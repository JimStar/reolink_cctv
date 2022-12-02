"""This component encapsulates the NVR/camera API and subscription."""

import logging
import os
import ssl
import datetime as dt
import aiohttp

from    typing                 import Optional
from    dateutil.relativedelta import relativedelta
from    xml.etree              import ElementTree as XML

import  homeassistant.util.dt                   as dt_util
from    homeassistant.core                      import HomeAssistant
from    homeassistant.helpers.network           import get_url, NoURLAvailableError
from    homeassistant.helpers.storage           import STORAGE_DIR
from    homeassistant.helpers.aiohttp_client    import async_create_clientsession
from    homeassistant.const                     import (
    CONF_HOST,
    CONF_PORT,
    CONF_USERNAME,
    CONF_PASSWORD,
    CONF_TIMEOUT
)

from reolink_ip.typings     import SearchTime
from reolink_ip.api         import (
    Host,
    MOTION_DETECTION_TYPE,
    FACE_DETECTION_TYPE,
    PERSON_DETECTION_TYPE,
    VEHICLE_DETECTION_TYPE,
    PET_DETECTION_TYPE,
    VISITOR_DETECTION_TYPE
)

from .const import (
    MOTION_COMMON_TYPE,
    CONF_PLAYBACK_DAYS,
    DEFAULT_PLAYBACK_DAYS,
    CONF_USE_HTTPS,
    CONF_MOTION_OFF_DELAY,
    CONF_MOTION_FORCE_OFF,
    CONF_PROTOCOL,
    CONF_STREAM,
    CONF_STREAM_FORMAT,
    CONF_SUBSCRIPTION_WATCHDOG_INTERVAL,
    DEFAULT_USE_HTTPS,
    DEFAULT_CHANNELS,
    DEFAULT_MOTION_OFF_DELAY,
    DEFAULT_MOTION_FORCE_OFF,
    DEFAULT_PROTOCOL,
    DEFAULT_STREAM,
    DEFAULT_STREAM_FORMAT,
    DEFAULT_TIMEOUT,
    DEFAULT_SUBSCRIPTION_WATCHDOG_INTERVAL,
    DOMAIN,
    SESSION_RENEW_THRESHOLD
)

_LOGGER         = logging.getLogger(__name__)
_LOGGER_DATA    = logging.getLogger(__name__ + ".data")

STORAGE_VERSION = 1


##########################################################################################################################################################
# Reolink Host class
##########################################################################################################################################################
class ReolinkHost:
    """The implementation of the Reolink Host class."""

    # Warning once in the logs that Internal URL is using HTTP while external URL is using HTTPS which is incompatible
    # with HomeAssistant starting 2022.3 when trying to retrieve internal URL
    warnedAboutNoURLAvailableError = False

    def __init__(self, hass: HomeAssistant, config: dict, options: Optional[dict] = {}):  # pylint: disable=too-many-arguments
        """Initialize Reolink Host. Could be either NVR, or Camera."""
        # global last_known_hass
        # last_known_hass = hass

        self._hass: HomeAssistant   = hass
        self.async_functions        = list()
        self.sync_functions         = list()

        from .camera import ReolinkCamera
        self.cameras: dict[int, ReolinkCamera] = dict()

        from .binary_sensor import MotionSensor, ObjectDetectedSensor, VisitorSensor

        self.sensor_motion_detection:   dict[int, MotionSensor]         = dict()
        self.sensor_face_detection:     dict[int, ObjectDetectedSensor] = dict()
        self.sensor_person_detection:   dict[int, ObjectDetectedSensor] = dict()
        self.sensor_vehicle_detection:  dict[int, ObjectDetectedSensor] = dict()
        self.sensor_pet_detection:      dict[int, ObjectDetectedSensor] = dict()
        self.sensor_visitor_detection:  dict[int, VisitorSensor]        = dict()

        self.motion_detection_enabled: Optional[list(bool)] = None

        use_https = DEFAULT_USE_HTTPS
        if CONF_USE_HTTPS in config:
            use_https = config[CONF_USE_HTTPS]
            if config[CONF_PORT] == 80 and use_https:
                _LOGGER.warning("Port 80 was chosen, \"Use HTTPS\" set back to False.")
                use_https = False

        self._clientSession: Optional[aiohttp.ClientSession] = None
        
        self._api = Host(
            config[CONF_HOST],
            config[CONF_PORT],
            config[CONF_USERNAME],
            config[CONF_PASSWORD],
            use_https = use_https,
            stream = (DEFAULT_STREAM if CONF_STREAM not in options else options[CONF_STREAM]),
            stream_format = (DEFAULT_STREAM_FORMAT if CONF_STREAM_FORMAT not in options else options[CONF_STREAM_FORMAT]),
            protocol = (DEFAULT_PROTOCOL if CONF_PROTOCOL not in options else options[CONF_PROTOCOL]),
            timeout = (DEFAULT_TIMEOUT if CONF_TIMEOUT not in options else options[CONF_TIMEOUT]),
            aiohttp_get_session_callback = self.get_iohttp_session
        )

        self._unique_id: Optional[str] = None

        self.motion_off_delay: int                  = DEFAULT_MOTION_OFF_DELAY if CONF_MOTION_OFF_DELAY not in options else options[CONF_MOTION_OFF_DELAY]
        self.motion_force_off: int                  = DEFAULT_MOTION_FORCE_OFF if CONF_MOTION_FORCE_OFF not in options else options[CONF_MOTION_FORCE_OFF]
        self.playback_days: int                     = DEFAULT_PLAYBACK_DAYS if CONF_PLAYBACK_DAYS not in options else options[CONF_PLAYBACK_DAYS]
        self._thumbnail_path: Optional[str]         = None
        self.subscription_watchdog_interval: int    = DEFAULT_SUBSCRIPTION_WATCHDOG_INTERVAL if CONF_SUBSCRIPTION_WATCHDOG_INTERVAL not in options else options[CONF_SUBSCRIPTION_WATCHDOG_INTERVAL]

        ##############################################################################
        # Web-hook subscription
        self._event_id      = None
        self._webhook_id    = None
        self._webhook_url   = None
    #endof __init__()


    ##############################################################################
    # Properties
    @property
    def unique_id(self):
        """Create the unique ID, base for all entities."""
        return self._unique_id

    @property
    def event_id(self):
        """Return the event ID string."""
        return self._event_id

    @property
    def api(self):
        """Return the API object."""
        return self._api

    @property
    def thumbnail_path(self):
        """ Thumbnail storage location """
        return self._thumbnail_path

    @thumbnail_path.setter
    def thumbnail_path(self, value):
        """ Set custom thumbnail path"""
        self._thumbnail_path = value


    ##############################################################################
    # Class methods

    async def init(self) -> bool:
        self._api.expire_session()

        if await self._api.get_host_data():
            #await self._api.is_admin()
            if self._api.mac_address is None:
                return False

            if not self._api.onvif_enabled or not self._api.rtmp_enabled or not self._api.rtsp_enabled:
                _LOGGER.info("ONVIF, RTMP or RTSP is disabled on %s, trying to enable it...", self._api.nvr_name)
                if not await self._api.set_net_port(enable_onvif = True, enable_rtmp = True, enable_rtsp = True):
                    _LOGGER.error("Unable to switch on ONVIF, RTMP and RTSP on %s. You need it to be ON to receive notifications and view camera streams.", self._api.nvr_name)

            self.motion_detection_enabled = {c: True for c in self._api.channels}

            if self._unique_id is None: # Don't change it on-the-fly after the entry-ID got already initialized with current value
                self._unique_id = self._api.mac_address.replace(":", "")

            # Would needed only if there is no camera name in a "DevInfo" section, but looks like it's there for cameras, at least for those I tested on...
            # if not await self._api.get_states():
            #     return False

            if not await self.register_webhook():
                return False

            if self._thumbnail_path is None:
                self._thumbnail_path = self._hass.config.path(f"{STORAGE_DIR}/{DOMAIN}/{self._unique_id}")

            return True
        else:
            return False
    #endof init()


    async def update_states(self) -> bool:
        """Call the API of the camera device to update the states."""
        return await self._api.get_states()
    #endof update_states()


    async def disconnect(self):
        """Disconnect from the API, so the connection will be released."""
        try:
            await self._api.unsubscribe_all()
        except Exception as e:
            err = str(e)
            if err:
                _LOGGER.error("Error while unsubscribing ONVIF events for %s: %s", self._api.nvr_name, err)
            else:
                _LOGGER.error("Unknown error while unsubscribing ONVIF events for %s.", self._api.nvr_name)

        try:
            await self._api.logout()
        except Exception as e:
            err = str(e)
            if err:
                _LOGGER.error("Error while logging out of %s: %s", self._api.nvr_name, err)
            else:
                _LOGGER.error("Unknown error while logging out of %s.", self._api.nvr_name)
    #endof disconnect()


    async def stop(self):
        """Disconnect the API and deregister the event listener."""
        await self.unregister_webhook()
        await self.disconnect()
        for func in self.async_functions:
            await func()
        for func in self.sync_functions:
            await self._hass.async_add_executor_job(func)
    #endof stop()


    def get_iohttp_session(self) -> Optional[aiohttp.ClientSession]:
        """Return the iohttp session."""

        if self._clientSession is None or self._clientSession.closed:
            context = ssl.create_default_context()
            context.set_ciphers("DEFAULT")
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

            self._clientSession = async_create_clientsession(self._hass, verify_ssl = False)
            self._clientSession.connector._ssl = context

        return self._clientSession
    #endof get_iohttp_session()


    #TODO: USELESS so far, because Reolink has not means to get a shot from a specific time of a recording.
    #      Thus all these thumbnails will be THE SAME current-time shots, which are nothing to do with a specific recording.
    '''
    async def store_vod_thumbnails(self, channel: int, start: Optional[dt.datetime] = None, end: Optional[dt.datetime] = None):
        """ Run search and store VoD thumbnails """

        current_time = dt_util.now()
        if end is None:
            end = current_time
        if start is None:
            start = dt.datetime.combine(end.date(), dt.time.min)
            if self.playback_days > 0:
                start -= relativedelta(days = int(self.playback_days))

        directory = os.path.join(self.thumbnail_path, f"{channel}")

        _, files = await self._api.request_vod_files(start, end)
        for file in files:
            start       = searchtime_to_datetime(file["StartTime"], end.tzinfo)
            event_id    = str(start.timestamp())

            thumbnail = os.path.join(directory, f"{event_id}.{THUMBNAIL_EXTENSION}")
            if not os.path.isfile(thumbnail):
                if not os.path.isdir(directory):
                    os.makedirs(directory)
                if channel in self.cameras:
                    service_data = {
                        ATTR_ENTITY_ID: self.cameras[channel].entity_id,
                        ATTR_FILENAME: thumbnail,
                    }
                    await self._hass.services.async_call(CAMERA_DOMAIN, SERVICE_SNAPSHOT, service_data, blocking = False)
    #endof store_vod_thumbnails()
    '''


    async def cleanup_vod_thumbnails(self, channel: int):
        """ Cleanup older thumbnail files """
        start = dt_util.now() - relativedelta(days = int(self.playback_days))

        start_date_timestamp = start.timestamp()
        directory = os.path.join(self.thumbnail_path, f"{channel}")
        if os.path.isdir(directory):
            for f in os.listdir(directory):
                f = os.path.join(directory, f)
                if os.stat(f).st_mtime < start_date_timestamp:
                    os.remove(f)
    #endof cleanup_vod_thumbnails()


    ######################################################################################################################################################
    # Web-hook subscription
    ######################################################################################################################################################

    async def subscribe(self) -> bool:
        """Subscribe to motion events and set the webhook as a callback."""
        if self._webhook_id is None:
            if not self.register_webhook():
                return False
        else:
            if self._api.subscribed:
                _LOGGER.debug("Host %s: is already subscribed to webhook %s.", self._api.host, self._webhook_url)
                return True

        if await self._api.subscribe(self._webhook_url):
            _LOGGER.info("Host %s: subscribed successfully to webhook %s.", self._api.host, self._webhook_url)
        else:
            _LOGGER.debug("Host %s: webhook subscription failed.", self._api.host)
            return False

        return True
    #endof subscribe()


    async def renew(self) -> bool:
        """Renew the subscription of the motion events (lease time is set to 15 minutes)."""

        if not self._api.subscribed:
            _LOGGER.debug("Host %s: requested to renew a non-existing Reolink subscription, trying to subscribe from scratch...", self._api.host)
            return await self.subscribe()

        timer = self._api.renewtimer
        if timer <= 0:
            _LOGGER.debug("Host %s: Reolink subscription expired, trying to subscribe again...", self._api.host)
            return await self._api.subscribe(self._webhook_url)
        elif timer <= SESSION_RENEW_THRESHOLD:
            if not await self._api.renew():
                _LOGGER.debug("Host %s: error renewing Reolink subscription, trying to subscribe again...", self._api.host)
                return await self._api.subscribe(self._webhook_url)
            else:
                _LOGGER.info("Host %s SUCCESSFULLY renewed Reolink subscription", self._api.host)

        return True
    #endof renew()


    async def register_webhook(self) -> bool:
        device_name: str = self.api.nvr_name
        if not device_name:
            _LOGGER.error("Error registering a webhook for %s:%s: the device name is empty.", self.api.host, self.api.port)
            self._event_id      = None
            self._webhook_id    = None
            self._webhook_url   = None
            return False

        no_spaces_name      = device_name.replace(" ", "_")
        self._webhook_id    = f"reolink_{no_spaces_name}_webhook"#self._hass.components.webhook.async_generate_id()
        self._event_id      = self._webhook_id
        try:
            self._hass.components.webhook.async_register(DOMAIN, self._event_id, self._webhook_id, handle_webhook)
        except ValueError:
            _LOGGER.debug("Error registering webhook %s. Trying to unregister it first and re-register again.", self._webhook_id)
            try:
                self._hass.components.webhook.async_unregister(self._webhook_id)
            except Exception as e:
                _LOGGER.debug("Error unregistering webhook %s: %s", self._webhook_id, str(e))

            try:
                self._hass.components.webhook.async_register(DOMAIN, self._event_id, self._webhook_id, handle_webhook)
            except ValueError:
                _LOGGER.error("Error registering a webhook %s for %s: maybe a duplicate device-name in your setup?", self._webhook_id, device_name)
                self._event_id      = None
                self._webhook_id    = None
                self._webhook_url   = None
                return False
            except Exception as e:
                err = str(e)
                if err:
                    _LOGGER.error("Error registering a webhook %s for %s: %s", self._webhook_id, device_name, err)
                else:
                    _LOGGER.error("Unknown error registering a webhook %s for %s.", self._webhook_id, device_name)
                self._event_id      = None
                self._webhook_id    = None
                self._webhook_url   = None
                return False
        except Exception as e:
            err = str(e)
            if err:
                _LOGGER.error("Error registering a webhook %s for %s: %s", self._webhook_id, device_name, err)
            else:
                _LOGGER.error("Unknown error registering a webhook %s for %s.", self._webhook_id, device_name)
            self._event_id      = None
            self._webhook_id    = None
            self._webhook_url   = None
            return False

        try:
            self._webhook_url = "{}{}".format(
                get_url(self._hass, prefer_external = False),
                self._hass.components.webhook.async_generate_path(self._webhook_id),
            )
        except NoURLAvailableError:
            if not warnedAboutNoURLAvailableError:
                warnedAboutNoURLAvailableError = True
                _LOGGER.warning("You're using HTTP for internal URL while using HTTPS for external URL in HA, which is not supported anymore by HomeAssistant starting 2022.3.\n"
                 "Please change your configuration to use HTTPS for internal URL or disable HTTPS for external.")
            try:
                self._webhook_url = "{}{}".format(
                    get_url(self._hass, prefer_external = True),
                    self._hass.components.webhook.async_generate_path(self._webhook_id),
                )
            except NoURLAvailableError:
                _LOGGER.error("Error registering URL for webhook %s: URL is not available.", self._webhook_id)
                try:
                    self._hass.components.webhook.async_unregister(self._webhook_id)
                except Exception as e:
                    _LOGGER.debug("Error unregistering webhook %s: %s", self._webhook_id, str(e))
                self._event_id      = None
                self._webhook_id    = None
                self._webhook_url   = None
                return False
            except Exception as e:
                err = str(e)
                if err:
                    _LOGGER.error("Error registering URL for webhook %s: %s", self._webhook_id, err)
                else:
                    _LOGGER.error("Unknown error registering URL for webhook %s.", self._webhook_id)
                try:
                    self._hass.components.webhook.async_unregister(self._webhook_id)
                except Exception as e:
                    _LOGGER.debug("Error unregistering webhook %s: %s", self._webhook_id, str(e))
                self._event_id      = None
                self._webhook_id    = None
                self._webhook_url   = None
                return False
        except Exception as e:
            err = str(e)
            if err:
                _LOGGER.error("Error registering URL for webhook %s: %s", self._webhook_id, err)
            else:
                _LOGGER.error("Unknown error registering URL for webhook %s.", self._webhook_id)
            try:
                self._hass.components.webhook.async_unregister(self._webhook_id)
            except Exception as e:
                _LOGGER.debug("Error unregistering webhook %s: %s", self._webhook_id, str(e))
            self._event_id      = None
            self._webhook_id    = None
            self._webhook_url   = None
            return False

        _LOGGER.info("Registered webhook: %s.", self._webhook_id)
        return True
    #endof register_webhook()


    async def unregister_webhook(self):
        """Unregister the webhook for motion events."""
        if self._webhook_id:
            _LOGGER.info("Unregistering webhook %s", self._webhook_id)
            try:
                self._hass.components.webhook.async_unregister(self._webhook_id)
            except Exception as e:
                _LOGGER.debug("Error unregistering webhook %s: %s", self._webhook_id, str(e))
        self._event_id      = None
        self._webhook_id    = None
        self._webhook_url   = None
    #endof unregister_webhook()
#endof class ReolinkHost


##########################################################################################################################################################
# GLOBALS
##########################################################################################################################################################

# Warning once in the logs that Internal URL is using HTTP while external URL is using HTTPS which is incompatible
# with HomeAssistant starting 2022.3 when trying to retrieve internal URL
#warnedAboutNoURLAvailableError = False
#last_known_hass: Optional[HomeAssistant] = None


async def handle_webhook(hass: HomeAssistant, webhook_id: str, request):
    """Handle incoming webhook from Reolink for inbound messages and calls."""

    _LOGGER.info("Webhook called (%s).", webhook_id)

    if not request.body_exists:
        _LOGGER.info("Webhook triggered without payload (%s).", webhook_id)

    data = await request.text()
    if not data:
        _LOGGER.info("Webhook triggered with unknown payload (%s).", webhook_id)
        return

    _LOGGER_DATA.debug("Webhook received payload (%s):\n%s", webhook_id, data)

    motion                      = None
    face                        = None
    person                      = None
    vehicle                     = None
    pet                         = None
    motion_alarm                = None
    visitor                     = None
    motion_common_notification  = True

    root = XML.fromstring(data)
    for message in root.iter('{http://docs.oasis-open.org/wsn/b-2}NotificationMessage'):
        topic_element = message.find("{http://docs.oasis-open.org/wsn/b-2}Topic[@Dialect='http://www.onvif.org/ver10/tev/topicExpression/ConcreteSet']")
        if topic_element is None:
            continue
        rule = os.path.basename(topic_element.text)
        if not rule:
            continue

        if rule == "Motion":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='IsMotion']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                motion = data_element.attrib["Value"] == "true"
        elif rule == "FaceDetect":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='State']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                face = data_element.attrib["Value"] == "true"
                if motion_common_notification:
                    motion_common_notification = False
        elif rule == "PeopleDetect":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='State']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                person = data_element.attrib["Value"] == "true"
                if motion_common_notification:
                    motion_common_notification = False
        elif rule == "VehicleDetect":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='State']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                vehicle = data_element.attrib["Value"] == "true"
                if motion_common_notification:
                    motion_common_notification = False
        elif rule == "DogCatDetect":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='State']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                pet = data_element.attrib["Value"] == "true"
                if motion_common_notification:
                    motion_common_notification = False
        elif rule == "MotionAlarm":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='State']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                motion_alarm = data_element.attrib["Value"] == "true"
        elif rule == "Visitor":
            data_element = message.find(".//{http://www.onvif.org/ver10/schema}SimpleItem[@Name='State']")
            if data_element is None:
                continue
            if "Value" in data_element.attrib:
                visitor = data_element.attrib["Value"] == "true"

    if motion is not None:
        if motion_common_notification:
            hass.bus.async_fire(webhook_id, {MOTION_COMMON_TYPE: motion})
        else:
            hass.bus.async_fire(webhook_id, {MOTION_DETECTION_TYPE: motion})
    elif motion_alarm is not None:
        if motion_common_notification:
            hass.bus.async_fire(webhook_id, {MOTION_COMMON_TYPE: motion_alarm})
        else:
            hass.bus.async_fire(webhook_id, {MOTION_DETECTION_TYPE: motion_alarm})
    if face is not None:
        hass.bus.async_fire(webhook_id, {FACE_DETECTION_TYPE: face})
    if person is not None:
        hass.bus.async_fire(webhook_id, {PERSON_DETECTION_TYPE: person})
    if vehicle is not None:
        hass.bus.async_fire(webhook_id, {VEHICLE_DETECTION_TYPE: vehicle})
    if pet is not None:
        hass.bus.async_fire(webhook_id, {PET_DETECTION_TYPE: pet})
    if visitor is not None:
        hass.bus.async_fire(webhook_id, {VISITOR_DETECTION_TYPE: visitor})
#endof handle_webhook()


def searchtime_to_datetime(self: SearchTime, timezone: dt.tzinfo):
    """ Convert SearchTime to datetime """
    return dt.datetime(
        self["year"],
        self["mon"],
        self["day"],
        self["hour"],
        self["min"],
        self["sec"],
        tzinfo = timezone,
    )
#endof searchtime_to_datetime()


# def callback_get_iohttp_session():
#     """Return the iohttp session for the last known hass instance."""
#     global last_known_hass
#     if last_known_hass is None:
#         raise Exception("No Home Assistant instance found")

#     context = ssl.create_default_context()
#     context.set_ciphers("DEFAULT")
#     context.check_hostname = False
#     context.verify_mode = ssl.CERT_NONE

#     session = async_create_clientsession(last_known_hass, verify_ssl = False)
#     session.connector._ssl = context

#     return session
# #endof callback_get_iohttp_session()

from typing import Optional

import gradio

from facefusion import config, translator
from facefusion.camera_manager import detect_local_camera_ids
from facefusion.common_helper import get_first
from facefusion.uis import choices as uis_choices
from facefusion.uis.core import register_ui_component

WEBCAM_DEVICE_ID_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_MODE_RADIO : Optional[gradio.Radio] = None
WEBCAM_RESOLUTION_DROPDOWN : Optional[gradio.Dropdown] = None
WEBCAM_FPS_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global WEBCAM_DEVICE_ID_DROPDOWN
	global WEBCAM_MODE_RADIO
	global WEBCAM_RESOLUTION_DROPDOWN
	global WEBCAM_FPS_SLIDER

	local_camera_ids = detect_local_camera_ids(0, 10) or [ 'none' ] #type:ignore[list-item]
	default_webcam_mode = config.get_str_value('webcam', 'webcam_mode', uis_choices.webcam_modes[0])
	default_webcam_resolution = config.get_str_value('webcam', 'webcam_resolution', uis_choices.webcam_resolutions[0])
	default_webcam_fps = config.get_int_value('webcam', 'webcam_fps', 30)

	WEBCAM_DEVICE_ID_DROPDOWN = gradio.Dropdown(
		value = get_first(local_camera_ids),
		label = translator.get('uis.webcam_device_id_dropdown'),
		choices = local_camera_ids
	)
	WEBCAM_MODE_RADIO = gradio.Radio(
		label = translator.get('uis.webcam_mode_radio'),
		choices = uis_choices.webcam_modes,
		value = default_webcam_mode if default_webcam_mode in uis_choices.webcam_modes else uis_choices.webcam_modes[0]
	)
	WEBCAM_RESOLUTION_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.webcam_resolution_dropdown'),
		choices = uis_choices.webcam_resolutions,
		value = default_webcam_resolution if default_webcam_resolution in uis_choices.webcam_resolutions else uis_choices.webcam_resolutions[0]
	)
	WEBCAM_FPS_SLIDER = gradio.Slider(
		label = translator.get('uis.webcam_fps_slider'),
		value = default_webcam_fps,
		step = 1,
		minimum = 1,
		maximum = 30
	)
	register_ui_component('webcam_device_id_dropdown', WEBCAM_DEVICE_ID_DROPDOWN)
	register_ui_component('webcam_mode_radio', WEBCAM_MODE_RADIO)
	register_ui_component('webcam_resolution_dropdown', WEBCAM_RESOLUTION_DROPDOWN)
	register_ui_component('webcam_fps_slider', WEBCAM_FPS_SLIDER)

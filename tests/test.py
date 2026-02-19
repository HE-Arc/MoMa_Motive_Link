import time

from MoMaMotiveLink.core import MotiveLink

if __name__ == "__main__":
    link = MotiveLink()
    link.start(use_multicast=False)
    while link.is_ready() is False:
        print("Waiting for MotiveLink to be ready...")
        time.sleep(1)

    skel = link.get_skeleton_definition()
    print(skel)

    print("Setting live mode...")
    result = link.set_live_mode()
    print(result)

    time.sleep(1)

    print("Setting edit mode...")
    result = link.set_edit_mode()
    print(result)

    time.sleep(1)

    print("Setting timeline play...")
    result = link.set_timeline_play()
    print(result)

    time.sleep(1)

    print("Setting timeline stop...")
    result = link.set_timeline_stop()
    print(result)

    time.sleep(1)

    print("Setting playback start frame...")
    result = link.set_playback_start_frame(1000)
    print(result)

    time.sleep(1)

    print("Setting playback stop frame...")
    result = link.set_playback_stop_frame(2000)
    print(result)

    time.sleep(1)

    print("Setting playback current frame...")
    result = link.set_playback_current_frame(1500)
    print(result)

    time.sleep(1)

    print("Setting playback looping off...")
    result = link.set_playback_looping(False)
    print(result)

    time.sleep(2)

    print("Setting timeline play...")
    result = link.set_timeline_play()
    print(result)

    time.sleep(1)

    print("Setting playback looping on...")
    result = link.set_playback_looping(True)
    print(result)

    time.sleep(1)

    print("Setting timeline play...")
    result = link.set_timeline_play()
    print(result)

    exit()
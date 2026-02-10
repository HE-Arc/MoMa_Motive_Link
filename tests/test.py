import time

from MoMaMotiveLink.core import MotiveLink

if __name__ == "__main__":
    anim_data = MotiveLink()
    anim_data.start()
    while anim_data.is_ready() is False:
        print("Waiting for MotiveLink to be ready...")
        time.sleep(1)

    skel = anim_data.get_skeleton_definition()
    print(skel)
    exit()
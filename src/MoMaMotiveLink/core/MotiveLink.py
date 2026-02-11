import os
import socket
import sys
import time

import numpy as np
from numpy._typing import NDArray

from MoMaMotiveLink.core import Tools
from MoMaMotiveLink.natnetsdk.DataDescriptions import DataDescriptions, SkeletonDescription, RigidBodyDescription
from MoMaMotiveLink.natnetsdk.MoCapData import MoCapData, SkeletonData, Skeleton, RigidBody
from MoMaMotiveLink.natnetsdk.NatNetClient import NatNetClient

from enum import Enum


class LINK_STATUS(Enum):
    WAIT = 1
    READY = 2


class MotiveLink:
    def __init__(self):
        self.status = LINK_STATUS.WAIT

        self.bone_id_to_name: dict[int, str] = {}
        self.bone_parents: NDArray[np.int32] = np.empty((0,), dtype=np.int32)  # np.array int32

        # --- Ajout : Stockage de la Bind Pose (Pose de repos) ---
        # Ces données définissent la forme du squelette sans animation
        self.rest_positions: NDArray[np.float64] = None  # (B, 3)
        self.rest_rotations: NDArray[np.float64] = None  # (B, 4) - Quaternions
        self.rest_scales: NDArray[np.float64] = None  # (B, 3)

        # Données d'animation
        self.local_matrices: NDArray[np.float64] = None  # (B, 4, 4) - Matrices de transformation complètes

    def start(self, use_multicast=False):
        print("This is the MotiveLink core module.")

        server_hostname = os.getenv("SERVER_HOSTNAME")
        try:
            motive_ip = socket.gethostbyname(server_hostname)
            print(f"Serveur Motive trouvé à l'adresse : {motive_ip}")

        except socket.gaierror:
            print(f"ERREUR : Impossible de trouver l'ordinateur nommé '{server_hostname}' sur le réseau.")
            exit()

        # Récupérer automatiquement MON adresse IP (celle du PC Python)
        hostname_local = socket.gethostname()
        client_ip = os.getenv("CLIENT_IP", None)
        if client_ip is None:
            client_ip = Tools.get_real_local_ip()

        print(f"Mon IP Client {hostname_local} est : {client_ip}")

        # Setup the client
        self.streamingClient = NatNetClient()
        self.streamingClient.set_print_level(0)

        # Configure the client to connect to Motive
        self.streamingClient.set_client_address(client_ip)  # Your IP (or 127.0.0.1 if local)
        self.streamingClient.set_server_address(motive_ip)  # Motive IP (or 127.0.0.1 if local)
        self.streamingClient.set_use_multicast(use_multicast)  # Must match Motive 'Transmission Type'

        # Configure the callbacks
        # streamingClient.new_frame_listener = receive_new_frame
        # self.streamingClient.new_frame_with_data_listener = self.receive_new_frame_with_data
        self.streamingClient.new_frame_with_data_listener = self.receive_frame_with_skeleton
        # streamingClient.rigid_body_listener = receive_rigid_body_frame
        self.streamingClient.model_description_listener = self.receive_model_descriptions

        # Start the connection
        # the argument 'd' sets the data transfer mode to 'data + command'. The other option is 'command only' ('c').
        is_running = self.streamingClient.run('d')

        if not is_running:
            print("ERROR: Could not start streaming client.")
            try:
                sys.exit(1)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        time.sleep(1)

        if self.streamingClient.connected() is False:
            print("ERROR: Could not connect properly.  Check that Motive streaming is on.")  # type: ignore  # noqa F501
            try:
                sys.exit(2)
            except SystemExit:
                print("...")
            finally:
                print("exiting")

        self.print_configuration(self.streamingClient)
        self.request_data_descriptions(self.streamingClient)

        print("Connected to Motive! Streaming data...")

    def is_ready(self):
        return self.status == LINK_STATUS.READY

    def print_configuration(self, natnet_client):
        natnet_client.refresh_configuration()
        print("Connection Configuration:")
        print("  Client:          %s" % natnet_client.local_ip_address)
        print("  Server:          %s" % natnet_client.server_ip_address)
        print("  Command Port:    %d" % natnet_client.command_port)
        print("  Data Port:       %d" % natnet_client.data_port)

        changeBitstreamString = "  Can Change Bitstream Version = "
        if natnet_client.use_multicast:
            print("  Using Multicast")
            print("  Multicast Group: %s" % natnet_client.multicast_address)
            changeBitstreamString += "false"
        else:
            print("  Using Unicast")
            changeBitstreamString += "true"

        # NatNet Server Info
        application_name = natnet_client.get_application_name()
        nat_net_requested_version = natnet_client.get_nat_net_requested_version()
        nat_net_version_server = natnet_client.get_nat_net_version_server()
        server_version = natnet_client.get_server_version()

        print("  NatNet Server Info")
        print("    Application Name %s" % (application_name))
        print("    MotiveVersion  %d %d %d %d" % (server_version[0], server_version[1], server_version[2],
                                                  server_version[3]))  # type: ignore  # noqa F501
        print("    NatNetVersion  %d %d %d %d" % (nat_net_version_server[0], nat_net_version_server[1],
                                                  nat_net_version_server[2],
                                                  nat_net_version_server[3]))  # type: ignore  # noqa F501
        print("  NatNet Bitstream Requested")
        print("    NatNetVersion  %d %d %d %d" % (nat_net_requested_version[0], nat_net_requested_version[1],
                                                  # type: ignore  # noqa F501
                                                  nat_net_requested_version[2],
                                                  nat_net_requested_version[3]))  # type: ignore  # noqa F501

        print(changeBitstreamString)
        # print("command_socket = %s" % (str(natnet_client.command_socket)))
        # print("data_socket    = %s" % (str(natnet_client.data_socket)))
        print("  PythonVersion    %s" % (sys.version))

    def receive_model_descriptions(self, data_descs: DataDescriptions):
        self.status = LINK_STATUS.WAIT

        self.bone_id_to_name = {}  # Reset
        bone_parents = []
        rest_positions = []
        rest_rotations = []

        # On parcourt les squelettes trouvés dans les descriptions
        skeleton: SkeletonDescription
        for skeleton in data_descs.skeleton_list:
            print(f"Squelette trouvé : {skeleton.name}")

            # Dans DataDescriptions.py, les os sont dans 'rigid_body_description_list'
            bone_desc: RigidBodyDescription
            for bone_desc in skeleton.rigid_body_description_list:
                # On remplit le dictionnaire : ID (int) -> Nom (str)
                decoded_name = bone_desc.sz_name.decode()
                self.bone_id_to_name[bone_desc.id_num] = decoded_name
                bone_parents.append(bone_desc.parent_id)
                rest_positions.append(bone_desc.pos)
                rest_rotations.append(bone_desc.rot)

                print(f"   Mapping : ID {bone_desc.id_num} -> {decoded_name}")

        self.bone_parents = np.array(bone_parents, dtype=np.int32)
        self.rest_positions = np.array(rest_positions, dtype=np.float64) * 100.0  # TODO Verify units (cm <-> m?)
        self.rest_rotations = np.array(rest_rotations, dtype=np.float64)
        self.rest_scales = np.full_like(self.rest_positions, fill_value=1.0, dtype=np.float64)

        self.status = LINK_STATUS.READY

    def receive_new_frame_with_data(self, data_dict):
        if self.status is not LINK_STATUS.READY:
            # On attend d'avoir reçu les descriptions pour traiter les frames
            return

        order_list = ["frameNumber", "markerSetCount", "unlabeledMarkersCount",  # type: ignore  # noqa F841
                      "rigidBodyCount", "skeletonCount", "labeledMarkerCount",
                      "timecode", "timecodeSub", "timestamp", "isRecording",
                      "trackedModelsChanged", "offset", "mocap_data"]
        mocap_data: MoCapData = data_dict["mocap_data"]
        skeleton_data: SkeletonData = mocap_data.skeleton_data
        skeleton: Skeleton = skeleton_data.skeleton_list[0]
        rigidbody: RigidBody = skeleton.rigid_body_list[0]

        dump_args = True
        if dump_args is True:
            out_string = "    "
            for key in data_dict:
                out_string += key + "= "
                if key in data_dict:
                    out_string += str(data_dict[key]) + " "
                out_string += "/"
            print(out_string)

    def receive_frame_with_skeleton(self, data_dict):
        # print("Received frame with skeleton data")

        # 1. Récupérer l'objet global
        if "mocap_data" not in data_dict:
            return
        mocap_data = data_dict["mocap_data"]

        matrices = []

        # 2. Vérifier s'il y a des squelettes
        if mocap_data.skeleton_data and mocap_data.skeleton_data.skeleton_list:
            # 3. Boucler sur chaque squelette (Actor 1, Actor 2...)
            for skeleton in mocap_data.skeleton_data.skeleton_list:
                # print(f"Squelette ID: {skeleton.id_num}")

                # 4. Boucler sur chaque OS du squelette
                # Dans le SDK, les os sont stockés comme une liste de RigidBodies
                for bone in skeleton.rigid_body_list:
                    # Voici les données vitales pour votre animation :
                    bone_id = bone.id_num  # L'ID unique de l'os (ex: Hips, LeftArm...)
                    position = np.array(bone.pos) * 100  # [x, y, z] # TODO Verify units (cm <-> m?)
                    rotation = np.array(bone.rot)  # [qx, qy, qz, qw] (Quaternion)
                    scale = np.array([1.0, 1.0, 1.0])  # Motive ne fournit pas d'échelle, on suppose 1.0

                    # position + rotation + scale into a 4x4 transform matrix
                    transform_matrix = Tools.compose_transform(position, rotation, scale)
                    matrices.append(transform_matrix)

        self.local_matrices = np.array(matrices, dtype=np.float64)
        # print(self.local_matrices)

    def get_skeleton_definition(self) -> dict:
        """
        Génère un dictionnaire contenant la structure statique du squelette.
        Idéal pour être envoyé en JSON au client lors de l'initialisation.
        """
        parents_list: list[int] = [-1]
        for parent in self.bone_parents:
            parent_name = self.bone_id_to_name.get(parent, None)
            if parent_name is None:
                continue
            id = list(self.bone_id_to_name.values()).index(parent_name)
            parents_list.append(id)

        # Préparation de la bind pose (si disponible, sinon identité)
        num_bones = len(self.bone_id_to_name)

        # Valeurs par défaut si les loaders n'ont pas rempli les rest_xxx
        r_pos = self.rest_positions.tolist() if self.rest_positions is not None else [[0, 0, 0]] * num_bones
        r_rot = self.rest_rotations.tolist() if self.rest_rotations is not None else [[0, 0, 0, 1]] * num_bones
        r_scl = self.rest_scales.tolist() if self.rest_scales is not None else [[1, 1, 1]] * num_bones

        return {
            "type": "SKELETON_DEF",
            "bone_names": list(self.bone_id_to_name.values()),
            "parents": parents_list,
            "bind_pose": {
                "positions": r_pos,
                "rotations": r_rot,
                "scales": r_scl
            }
        }

    def request_data_descriptions(self, s_client):
        # Request the model definitions
        return s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF, "",
                                     (s_client.server_ip_address, s_client.command_port))

    # region NATNET COMMANDS
    # * Find the command list here : https://docs.optitrack.com/developer-tools/natnet-sdk/natnet-remote-requests-commands
    def set_live_mode(self) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command("LiveMode")

    def set_edit_mode(self) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command("EditMode")

    # region PLAYBACK AND TIMELINE CONTROLS
    def set_timeline_play(self) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command("TimelinePlay")

    def set_timeline_stop(self) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command("TimelineStop")

    def set_playback_take_name(self, playback_take_name: str) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command(f"SetPlaybackTakeName,{playback_take_name}")

    def set_playback_start_frame(self, start_frame_number: int) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command(f"SetPlaybackStartFrame,{start_frame_number}")

    def set_playback_stop_frame(self, stop_frame_number: int) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command(f"SetPlaybackStopFrame,{stop_frame_number}")

    # TODO : This is not working as intended
    def set_playback_current_frame(self, current_frame_number: int) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command(f"SetPlaybackCurrentFrame,{current_frame_number}")

    def set_playback_looping(self, loop: bool) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        if loop:
            return self.streamingClient.send_command("SetPlaybackLooping")
        else:
            return self.streamingClient.send_command("SetPlaybackLooping,0")

    # endregion

    # region RECORDING COMMANDS
    def set_recording_start(self) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command("StartRecording")

    def set_recording_stop(self) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command("StopRecording")

    def set_record_take_name(self, record_take_name: str) -> int:
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command(f"SetRecordTakeName,{record_take_name}")

    def set_current_session(self, session_name: str) -> int:
        """
        Set current session. If the session name already exists, Motive switches to that session.
        If the session does not exist, Motive will create a new session.
        You can use absolute paths to define folder locations.
        :param session_name: Name or absolute path of the session
        :return:
        """
        if self.streamingClient is None or self.status is not LINK_STATUS.READY:
            return -1

        return self.streamingClient.send_command(f"SetCurrentSession,{session_name}")

    # endregion
    # endregion

    def dispose(self):
        self.status = LINK_STATUS.WAIT
        if self.streamingClient is not None:
            self.streamingClient.shutdown()


if __name__ == "__main__":
    motive_link = MotiveLink()
    motive_link.start()

    # Keep the main thread alive to let the listener thread work
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        motive_link.dispose()

    exit()

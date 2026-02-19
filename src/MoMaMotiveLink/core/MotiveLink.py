import logging
import os
import socket
import time
from enum import Enum

import numpy as np
from natnet import DataDescriptions, DataFrame, NatNetClient, SkeletonDescription, RigidBodyDescription
from numpy._typing import NDArray

from MoMaMotiveLink.core import Tools

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger("MoMaMotiveLinkLogger")
logger.setLevel(logging.INFO)


class LINK_STATUS(Enum):
    STARTING = 0
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

    def set_log_level(self, level: int):
        logger.setLevel(level)
        logger.info(f"MoMaMotiveLink log level set to {logging.getLevelName(level)}")

    def start(self, use_multicast=False):
        logger.info("This is the MotiveLink core module.")

        server_hostname = os.getenv("MOMA_MOTIVELINK_SERVER_HOSTNAME")
        client_ip = os.getenv("MOMA_MOTIVELINK_CLIENT_IP", None)

        logger.info(f"MOMA_MOTIVELINK_SERVER_HOSTNAME: {server_hostname}", )
        logger.info(f"MOMA_MOTIVELINK_CLIENT_IP: {client_ip}")

        try:
            motive_ip = socket.gethostbyname(server_hostname)
            logger.info(f"Serveur Motive trouvé à l'adresse : {motive_ip}")

        except socket.gaierror:
            logger.info(f"ERREUR : Impossible de trouver l'ordinateur nommé '{server_hostname}' sur le réseau.")
            exit()

        # Récupérer automatiquement MON adresse IP (celle du PC Python)
        hostname_local = socket.gethostname()
        if client_ip is None:
            client_ip = Tools.get_real_local_ip()

        logger.info(f"Mon IP Client {hostname_local} est : {client_ip}")

        # Setup the client
        self.streaming_client = NatNetClient(server_ip_address=motive_ip, local_ip_address=client_ip,
                                             use_multicast=use_multicast)
        self.streaming_client.on_data_description_received_event.handlers.append(self.receive_model_descriptions)
        self.streaming_client.on_data_frame_received_event.handlers.append(self.receive_frame_with_skeleton)

        self.streaming_client.connect()
        self.streaming_client.request_modeldef()
        time.sleep(1)
        self.streaming_client.run_async()

        logger.info("MotiveLink started and listening for data...")

    def is_ready(self):
        return self.status == LINK_STATUS.READY

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

    def receive_model_descriptions(self, data_descs: DataDescriptions):
        logger.debug("Received model descriptions from Motive.")

        self.status = LINK_STATUS.WAIT

        self.bone_id_to_name = {}  # Reset
        bone_parents = []
        rest_positions = []
        rest_rotations = []

        # On parcourt les squelettes trouvés dans les descriptions
        skeleton: SkeletonDescription
        for skeleton in data_descs.skeletons:
            print(f"Squelette trouvé : {skeleton.name}")

            # Dans DataDescriptions.py, les os sont dans 'rigid_body_description_list'
            bone_desc: RigidBodyDescription
            for bone_desc in skeleton.rigid_body_descriptions:
                # On remplit le dictionnaire : ID (int) -> Nom (str)
                self.bone_id_to_name[bone_desc.id_num] = bone_desc.name
                bone_parents.append(bone_desc.parent_id)
                rest_positions.append(bone_desc.pos)
                rest_rotations.append(bone_desc.quat)

                print(f"   Mapping : ID {bone_desc.id_num} -> {bone_desc.name}")

        self.bone_parents = np.array(bone_parents, dtype=np.int32)
        self.rest_positions = np.array(rest_positions, dtype=np.float64) * 100.0  # TODO Verify units (cm <-> m?)
        self.rest_rotations = np.array(rest_rotations, dtype=np.float64)
        self.rest_scales = np.full_like(self.rest_positions, fill_value=1.0, dtype=np.float64)

        self.status = LINK_STATUS.READY

    def receive_frame_with_skeleton(self, dataframe: DataFrame):
        if self.status is not LINK_STATUS.READY:
            # On attend d'avoir reçu les descriptions pour traiter les frames
            return

        logger.debug("Received frame with skeleton data")

        matrices = []

        # 2. Vérifier s'il y a des squelettes
        if dataframe.skeletons:
            # 3. Boucler sur chaque squelette (Actor 1, Actor 2...)
            for skeleton in dataframe.skeletons:
                # print(f"Squelette ID: {skeleton.id_num}")

                # 4. Boucler sur chaque OS du squelette
                # Dans le SDK, les os sont stockés comme une liste de RigidBodies
                for bone in skeleton.rigid_bodies:
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

    def dispose(self):
        self.status = LINK_STATUS.WAIT
        if self.streaming_client is not None:
            self.streaming_client.shutdown()


if __name__ == "__main__":
    motive_link = MotiveLink()
    motive_link.set_log_level(logging.DEBUG)
    motive_link.start(use_multicast=False)

    # Keep the main thread alive to let the listener thread work
    try:
        while True:
            time.sleep(1)
            # motive_link.streaming_client.update_sync()
    except KeyboardInterrupt:
        print("Stopping...")
        motive_link.dispose()

    exit()

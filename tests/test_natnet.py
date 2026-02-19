import logging
import os
import socket
import time

from natnet import DataDescriptions, DataFrame, NatNetClient

from MoMaMotiveLink.core import Tools

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s: %(message)s')
logger = logging.getLogger("MoMaMotiveLinkLogger")
logger.setLevel(logging.INFO)

def receive_new_frame(data_frame: DataFrame):
    # print("Received new frame.")
    global num_frames
    num_frames += 1


def receive_new_desc(desc: DataDescriptions):
    print("Received data descriptions.")


num_frames = 0
if __name__ == "__main__":

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

    streaming_client = NatNetClient(server_ip_address=motive_ip, local_ip_address=client_ip, use_multicast=False)
    streaming_client.on_data_description_received_event.handlers.append(receive_new_desc)
    streaming_client.on_data_frame_received_event.handlers.append(receive_new_frame)

    # with streaming_client:
    streaming_client.connect()
    # Wait for the client to connect and retrieve server info (including version)
    # time.sleep(1)
    # Ensure we process any initial packets (like ServerInfo) to update protocol version
    # streaming_client.update_sync()

    streaming_client.request_modeldef()
    time.sleep(1)
    # streaming_client.update_sync()

    streaming_client.run_async()

    for i in range(10):
        time.sleep(1)
        # streaming_client.update_sync()
        print(f"Received {num_frames} frames in {i + 1}s")

    streaming_client.shutdown()

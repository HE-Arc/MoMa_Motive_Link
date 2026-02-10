import socket

import numpy as np


def get_real_local_ip():
    """
    Determines the local IP address of the machine.

    This function creates a UDP socket to simulate a connection to an external server (Google's public DNS server at 8.8.8.8).
    It retrieves the local IP address used for this connection. If an error occurs (e.g., no network connectivity),
    it falls back to the loopback address (127.0.0.1).

    Returns:
        str: The local IP address of the machine, or '127.0.0.1' if an exception occurs.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  # Create a UDP socket on IPv4
    try:
        # Attempt to connect to an external server (address doesn't need to exist)
        s.connect(('8.8.8.8', 1))
        ip = s.getsockname()[0]  # Retrieve the local IP address
    except Exception:
        ip = '127.0.0.1'  # Fallback to loopback address in case of an error
    finally:
        s.close()  # Ensure the socket is closed to release resources
    return ip


def compose_transform(position, rotation, scale):
    output = np.zeros((4, 4), dtype=np.float64)

    # --- 1. Calcul des variables locales (Registres uniquement) ---
    qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
    sx, sy, sz = scale[0], scale[1], scale[2]
    px, py, pz = position[0], position[1], position[2]

    # Pré-calculs quaternion
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz

    # Définition des coefficients de la matrice locale (l_row_col)
    l00 = (1.0 - 2.0 * (yy + zz)) * sx
    l01 = (2.0 * (xy - wz)) * sy
    l02 = (2.0 * (xz + wy)) * sz
    l03 = px

    l10 = (2.0 * (xy + wz)) * sx
    l11 = (1.0 - 2.0 * (xx + zz)) * sy
    l12 = (2.0 * (yz - wx)) * sz
    l13 = py

    l20 = (2.0 * (xz - wy)) * sx
    l21 = (2.0 * (yz + wx)) * sy
    l22 = (1.0 - 2.0 * (xx + yy)) * sz
    l23 = pz

    output[0, 0], output[0, 1], output[0, 2], output[0, 3] = (
        l00,
        l01,
        l02,
        l03,
    )
    output[1, 0], output[1, 1], output[1, 2], output[1, 3] = (
        l10,
        l11,
        l12,
        l13,
    )
    output[2, 0], output[2, 1], output[2, 2], output[2, 3] = (
        l20,
        l21,
        l22,
        l23,
    )
    output[3, 0], output[3, 1], output[3, 2], output[3, 3] = (
        0.0,
        0.0,
        0.0,
        1.0,
    )

    return output
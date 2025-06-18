import requests

# If you set up the SSH tunnel as above, use localhost
url = "http://localhost:5000/estimate"

# Point these paths at your local copies of the images
path_target  = "/home/chifeng/OVMM/target.jpg"
path_current = "/home/chifeng/OVMM/current1.jpg"

files = {
    "target":  ("target.jpg", open(path_target,  "rb"), "image/jpeg"),
    "current": ("current.jpg", open(path_current, "rb"), "image/jpeg"),
}

try:
    resp = requests.post(url, files=files, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    print("ρ (rho):", data["rho"])
    print("φ (phi):", data["phi"])
except requests.exceptions.RequestException as e:
    print("Request failed:", e)


"""
ssh -N -L 5000:localhost:5000 cl6933@128.238.176.21
"""
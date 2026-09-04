import importlib.util
import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest


SAMPLER_PATH = Path(__file__).parents[1] / "humanoid_xml_sampler.py"
SPEC = importlib.util.spec_from_file_location("humanoid_xml_sampler", SAMPLER_PATH)
sampler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampler)


def _joint(root, name):
    return root.find(f".//worldbody//joint[@name='{name}']")


def _vector(element, attribute):
    return np.fromstring(element.get(attribute), sep=" ")


@pytest.mark.parametrize("noise_scale", [-0.1, 1.1, float("nan"), True, "1"])
def test_noise_scale_must_be_bounded(noise_scale):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        sampler.create_random_humanoid(noise_scale)


def test_zero_noise_matches_neutral_humanoid_parameters():
    random.seed(1)
    first = sampler.create_random_humanoid(noise_scale=0)
    random.seed(999)
    second = sampler.create_random_humanoid(noise_scale=0)
    assert first == second

    root = ET.fromstring(first)
    torso = root.find(".//body[@name='torso']")
    root_joint = _joint(root, "root")
    assert float(torso.get("pos").split()[2]) == pytest.approx(1.397826)
    assert root_joint.get("type") == "free"
    assert root_joint.get("limited") == "false"
    assert root_joint.get("axis") is None
    assert _joint(root, "right_knee").get("axis") == "0 -1 0"
    assert _joint(root, "right_hip_z").get("range") == "-60.0 35.0"

    motors = root.findall("./actuator/motor")
    assert [motor.get("name") for motor in motors][:7] == [
        "abdomen_y",
        "abdomen_z",
        "abdomen_x",
        "right_hip_x",
        "right_hip_z",
        "right_hip_y",
        "right_knee",
    ]
    assert len(motors) == 17
    assert len(root.findall(".//worldbody//body")) == 13


def test_full_noise_samples_remain_in_safe_ranges():
    base_gears = {
        "abdomen_x": 100.0,
        "abdomen_y": 100.0,
        "abdomen_z": 100.0,
        "hip_x": 100.0,
        "hip_z": 100.0,
        "hip_y": 300.0,
        "knee": 200.0,
        "shoulder1": 25.0,
        "shoulder2": 25.0,
        "elbow": 25.0,
    }
    for seed in range(100):
        random.seed(seed)
        root = ET.fromstring(sampler.create_random_humanoid(noise_scale=1))
        assert len(root.findall(".//worldbody//body")) == 13
        assert len(root.findall("./actuator/motor")) == 17

        torso_height = _vector(
            root.find(".//body[@name='torso']"), "pos"
        )[2]
        assert 1.25 < torso_height < 1.56
        friction = _vector(root.find(".//geom[@name='floor']"), "friction")[0]
        assert 0.7 <= friction <= 1.3

        thigh = root.find(".//geom[@name='right_thigh1']")
        thigh_radius = _vector(thigh, "size")[0]
        thigh_length = np.linalg.norm(
            _vector(thigh, "fromto")[3:] - _vector(thigh, "fromto")[:3]
        )
        assert 0.0552 <= thigh_radius <= 0.0648
        assert 0.2992 <= thigh_length <= 0.3808

        gears = {
            motor.get("name"): float(motor.get("gear"))
            for motor in root.findall("./actuator/motor")
        }
        for joint_name, value in gears.items():
            key = joint_name.removeprefix("left_").removeprefix("right_")
            assert 0.8 * base_gears[key] <= value <= 1.2 * base_gears[key]
        for name in ("hip_x", "hip_z", "hip_y", "knee", "elbow"):
            assert gears[f"left_{name}"] == gears[f"right_{name}"]

        for name in (
            "hip_x",
            "hip_z",
            "hip_y",
            "knee",
            "shoulder1",
            "shoulder2",
            "elbow",
        ):
            left = _joint(root, f"left_{name}")
            right = _joint(root, f"right_{name}")
            if name.startswith("shoulder"):
                left_range = _vector(left, "range")
                right_range = _vector(right, "range")
                assert right_range == pytest.approx(-left_range[::-1])
                for attribute in ("armature", "damping", "stiffness"):
                    assert left.get(attribute) == right.get(attribute)
            else:
                for attribute in (
                    "armature",
                    "damping",
                    "stiffness",
                    "range",
                ):
                    assert left.get(attribute) == right.get(attribute)


def test_random_models_compile_and_start_without_contacts():
    mujoco = pytest.importorskip("mujoco")
    for seed in range(25):
        random.seed(seed)
        xml = sampler.create_random_humanoid(noise_scale=1)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        assert (model.nq, model.nv, model.nu, model.nbody) == (24, 23, 17, 14)
        assert data.ncon == 0
        assert np.isfinite(data.qpos).all()

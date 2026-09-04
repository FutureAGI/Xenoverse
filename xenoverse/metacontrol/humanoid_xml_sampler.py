import math
import numbers
import random
import xml.etree.ElementTree as ET
from copy import deepcopy
from xml.dom import minidom

import numpy


# Full-scale randomization is deliberately conservative. MuJoCo infers mass
# and inertia from geometry, so a 2x radius change has a much larger effect on
# dynamics than the number itself suggests.
MAX_LENGTH_VARIATION = 0.12
MAX_RADIUS_VARIATION = 0.08
MAX_DYNAMICS_VARIATION = 0.20
MAX_JOINT_RANGE_VARIATION = 0.10


def _validate_noise_scale(noise_scale):
    if (
        isinstance(noise_scale, bool)
        or not isinstance(noise_scale, numbers.Real)
        or not math.isfinite(noise_scale)
        or not 0.0 <= noise_scale <= 1.0
    ):
        raise ValueError("noise_scale must be a finite number in [0, 1]")
    return float(noise_scale)


def _sample_relative(base, max_delta, noise_scale):
    delta = max_delta * noise_scale
    return base * random.uniform(1.0 - delta, 1.0 + delta)


def _sample_absolute(base, max_delta, noise_scale):
    delta = max_delta * noise_scale
    return random.uniform(base - delta, base + delta)


def default_rng(range_tuple, default=0):
    if isinstance(range_tuple, (int, float)):
        return range_tuple
    if isinstance(range_tuple, (list, tuple)) and len(range_tuple) == 2:
        return random.uniform(*range_tuple)
    return default


def sample_joint_attributes(
    armature,
    damping,
    stiffness,
    joint_range,
    noise_scale=1.0,
    joint_type="hinge",
):
    """Sample a joint near the parameters of Gymnasium's Humanoid."""
    noise_scale = _validate_noise_scale(noise_scale)
    lower, upper = joint_range
    range_delta = MAX_JOINT_RANGE_VARIATION * noise_scale
    lower = random.uniform(
        lower - abs(lower) * range_delta,
        lower + abs(lower) * range_delta,
    )
    upper = random.uniform(
        upper - abs(upper) * range_delta,
        upper + abs(upper) * range_delta,
    )
    return {
        "armature": _sample_relative(
            armature, MAX_DYNAMICS_VARIATION, noise_scale
        ),
        "damping": _sample_relative(damping, MAX_DYNAMICS_VARIATION, noise_scale),
        "stiffness": _sample_relative(
            stiffness, MAX_DYNAMICS_VARIATION, noise_scale
        ),
        "type": joint_type,
        "range": (lower, upper),
    }


def sample_joints_gear(parent, joint_names, noise_scale=1.0):
    """Create motors with paired, moderately randomized actuator strengths."""
    noise_scale = _validate_noise_scale(noise_scale)
    base_gears = {
        "abdomen_x": 100.0,
        "abdomen_y": 100.0,
        "abdomen_z": 100.0,
        "knee": 200.0,
        "hip_x": 100.0,
        "hip_z": 100.0,
        "hip_y": 300.0,
        "shoulder1": 25.0,
        "shoulder2": 25.0,
        "elbow": 25.0,
    }
    sampled_gears = {
        name: _sample_relative(value, MAX_DYNAMICS_VARIATION, noise_scale)
        for name, value in base_gears.items()
    }
    for joint_name in joint_names:
        joint_key = joint_name.removeprefix("left_").removeprefix("right_")
        ET_sub(
            parent,
            "motor",
            {
                "joint": joint_name,
                "gear": sampled_gears[joint_key],
                "name": joint_name,
            },
        )
    return sampled_gears


def strval(x):
    if isinstance(x, bool):
        return str(x).lower()
    if isinstance(x, (float, numpy.floating)):
        return str(round(float(x), 6))
    return str(x)


def ET_sub(parent_element, tag, attrib=None, text=None):
    """Create a subelement while serializing scalar and vector attributes."""
    new_attrib = {}
    for key, value in (attrib or {}).items():
        if value is None:
            new_attrib[key] = ""
        elif isinstance(value, (list, tuple, numpy.ndarray)):
            new_attrib[key] = " ".join(map(strval, value))
        else:
            new_attrib[key] = strval(value)
    element = ET.SubElement(parent_element, tag, new_attrib)
    if text is not None:
        element.text = text
    return element


def ET_set(element, attrib=None):
    """Set scalar and vector XML attributes on an element."""
    for key, value in (attrib or {}).items():
        if value is None:
            element.set(key, "")
        elif isinstance(value, (list, tuple, numpy.ndarray)):
            element.set(key, " ".join(map(strval, value)))
        else:
            element.set(key, strval(value))
    return element


def sample_all_joint_attributes(asymmetric=False, noise_scale=1.0):
    """Sample bounded joint dynamics and limits for the whole humanoid.

    Neutral values follow Gymnasium's Humanoid. Paired limbs share dynamics by
    default so random tasks do not have an unintended constant turning bias.
    """
    noise_scale = _validate_noise_scale(noise_scale)
    specs = {
        "abdomen_z": (0.02, 5.0, 20.0, (-45.0, 45.0)),
        "abdomen_y": (0.02, 5.0, 10.0, (-75.0, 30.0)),
        "abdomen_x": (0.02, 5.0, 10.0, (-35.0, 35.0)),
        "hip_x": (0.01, 5.0, 10.0, (-25.0, 5.0)),
        "hip_z": (0.01, 5.0, 10.0, (-60.0, 35.0)),
        "hip_y": (0.01, 5.0, 20.0, (-110.0, 20.0)),
        "knee": (0.006, 5.0, 1.0, (-160.0, -2.0)),
        "shoulder1": (0.0068, 5.0, 1.0, (-60.0, 85.0)),
        "shoulder2": (0.0051, 5.0, 1.0, (-60.0, 85.0)),
        "elbow": (0.0028, 5.0, 0.0, (-90.0, 50.0)),
    }
    attrs = {
        name: sample_joint_attributes(*spec, noise_scale=noise_scale)
        for name, spec in specs.items()
        if name.startswith("abdomen")
    }
    for name in ("hip_x", "hip_z", "hip_y", "knee", "elbow"):
        left = sample_joint_attributes(*specs[name], noise_scale=noise_scale)
        right = (
            sample_joint_attributes(*specs[name], noise_scale=noise_scale)
            if asymmetric
            else deepcopy(left)
        )
        attrs[f"left_{name}"] = left
        attrs[f"right_{name}"] = right

    for name in ("shoulder1", "shoulder2"):
        left = sample_joint_attributes(*specs[name], noise_scale=noise_scale)
        if asymmetric:
            right = sample_joint_attributes(
                specs[name][0],
                specs[name][1],
                specs[name][2],
                (-85.0, 60.0),
                noise_scale=noise_scale,
            )
        else:
            right = deepcopy(left)
            left_lower, left_upper = left["range"]
            right["range"] = (-left_upper, -left_lower)
        attrs[f"left_{name}"] = left
        attrs[f"right_{name}"] = right
    return attrs


def sample_all_limb_sizes(noise_scale=1.0):
    """Sample a coherent morphology around the standard Humanoid body."""
    noise_scale = _validate_noise_scale(noise_scale)
    torso_length_scale = _sample_relative(
        1.0, MAX_LENGTH_VARIATION * 0.67, noise_scale
    )
    leg_length_scale = _sample_relative(1.0, MAX_LENGTH_VARIATION, noise_scale)
    arm_length_scale = _sample_relative(1.0, MAX_LENGTH_VARIATION, noise_scale)
    torso_radius_scale = _sample_relative(
        1.0, MAX_RADIUS_VARIATION, noise_scale
    )
    leg_radius_scale = _sample_relative(1.0, MAX_RADIUS_VARIATION, noise_scale)
    arm_radius_scale = _sample_relative(1.0, MAX_RADIUS_VARIATION, noise_scale)

    sizes = {
        "size_head": 0.09 * torso_radius_scale,
        "size_torso1": 0.07 * torso_radius_scale,
        "len_torso1": 0.14 * torso_length_scale,
        "size_uwaist": 0.06 * torso_radius_scale,
        "len_uwaist": 0.12 * torso_length_scale,
        "dx_uwaist": _sample_absolute(-0.01, 0.008, noise_scale),
        "size_lwaist": 0.06 * torso_radius_scale,
        "dx_lwaist": _sample_absolute(-0.01, 0.008, noise_scale),
        "len_lwaist": 0.12 * torso_length_scale,
        "size_pelvis": 0.09 * torso_radius_scale,
        "len_pelvis": 0.14 * torso_length_scale,
        "size_thigh": 0.06 * leg_radius_scale,
        "len_thigh": 0.34 * leg_length_scale,
        "size_shin": 0.049 * leg_radius_scale,
        "len_shin": 0.30 * leg_length_scale,
        "size_foot": 0.075 * leg_radius_scale,
        "size_upper_arm": 0.04 * arm_radius_scale,
        # The official XML uses (0.16, 0.16, -0.16) as a from-to vector.
        "len_upper_arm": math.sqrt(3.0) * 0.16 * arm_length_scale,
        "size_lower_arm": 0.031 * arm_radius_scale,
        "len_lower_arm": math.sqrt(3.0) * 0.16 * arm_length_scale,
        "size_hand": 0.04 * arm_radius_scale,
    }
    sizes.update(
        {
            "t2h": 0.19 * torso_length_scale,
            "u2t": 0.12 * torso_length_scale,
            "l2t": 0.26 * torso_length_scale,
            "p2l": 0.165 * torso_length_scale,
            "th2p_x": _sample_absolute(0.0, 0.01, noise_scale),
            "th2p_y": 0.10 * torso_length_scale,
            "th2p_z": 0.04 * torso_length_scale,
            "s2th": sizes["len_thigh"]
            + sizes["size_thigh"]
            + 0.003 * leg_length_scale,
            # Distance from the shin origin to the centre of the foot.
            "f2s": sizes["len_shin"]
            + sizes["size_shin"]
            + 0.001 * leg_length_scale,
            "foot_height": sizes["size_foot"],
            "ua2t": 0.06 * torso_length_scale,
            "ua2t_y": 0.17 * torso_length_scale,
            "la2ua": math.sqrt(3.0) * 0.18 * arm_length_scale,
            "h2la": math.sqrt(3.0) * 0.18 * arm_length_scale,
            "la_origin": 0.01 * arm_length_scale,
            "lwaist_joint_pos": 0.065 * torso_length_scale,
            "pelvis_lwaist_joint_pos": 0.10 * torso_length_scale,
            "knee_joint_pos": 0.02 * leg_length_scale,
        }
    )
    lateral_slope = _sample_absolute(0.0294, 0.01, noise_scale)
    sizes["lthigh_direction"] = (0.0, -lateral_slope, -1.0)
    vertical_factor = 1.0 / math.sqrt(1.0 + lateral_slope**2)
    lateral_factor = lateral_slope / math.sqrt(1.0 + lateral_slope**2)
    # Independent torso/leg scaling must not let the spherical feet overlap.
    sizes["th2p_y"] = max(
        sizes["th2p_y"],
        lateral_factor * sizes["s2th"] + sizes["size_foot"] + 0.01,
    )
    ground_clearance = sizes["size_foot"] + 0.03 * leg_length_scale
    sizes["torso_height"] = (
        sizes["l2t"]
        + sizes["p2l"]
        + sizes["th2p_z"]
        + vertical_factor * sizes["s2th"]
        + sizes["f2s"]
        + sizes["size_foot"]
        + ground_clearance
    )
    return sizes


def prepare_assets(root):
    ET.SubElement(root, "compiler", {"angle": "degree", "inertiafromgeom": "true"})
    default = ET.SubElement(root, "default")
    ET.SubElement(
        default, "joint", {"armature": "1", "damping": "1", "limited": "true"}
    )
    ET.SubElement(
        default,
        "geom",
        {
            "conaffinity": "1",
            "condim": "1",
            "contype": "1",
            "margin": "0.001",
            "material": "geom",
            "rgba": "0.8 0.6 .4 1",
        },
    )
    ET.SubElement(
        default, "motor", {"ctrllimited": "true", "ctrlrange": "-.4 .4"}
    )
    ET.SubElement(
        root,
        "option",
        {
            "integrator": "RK4",
            "iterations": "50",
            "solver": "PGS",
            "timestep": "0.003",
        },
    )
    ET.SubElement(root, "size", {"nkey": "5", "nuser_geom": "1"})
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "map", {"fogend": "5", "fogstart": "3"})
    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "builtin": "gradient",
            "height": "100",
            "rgb1": ".4 .5 .6",
            "rgb2": "0 0 0",
            "type": "skybox",
            "width": "100",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "builtin": "flat",
            "height": "1278",
            "mark": "cross",
            "markrgb": "1 1 1",
            "name": "texgeom",
            "random": "0.01",
            "rgb1": "0.8 0.6 0.4",
            "rgb2": "0.8 0.6 0.4",
            "type": "cube",
            "width": "127",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "builtin": "checker",
            "height": "100",
            "name": "texplane",
            "rgb1": "0 0 0",
            "rgb2": "0.8 0.8 0.8",
            "type": "2d",
            "width": "100",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "MatPlane",
            "reflectance": "0.5",
            "shininess": "1",
            "specular": "1",
            "texrepeat": "60 60",
            "texture": "texplane",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {"name": "geom", "texture": "texgeom", "texuniform": "true"},
    )


def create_joint(
    parent_element,
    name,
    axis=(0, 1, 0),
    type="hinge",
    joint_attrs=None,
    **kwargs,
):
    """Create a joint element and attach it to the parent element."""
    attrib = {"name": name, "type": type}
    if type != "free":
        attrib["axis"] = axis
    joint = ET_sub(parent_element, "joint", attrib=attrib)
    if joint_attrs and name in joint_attrs:
        kwargs.update(joint_attrs[name])
    if type == "free":
        kwargs["limited"] = False
    return ET_set(joint, kwargs)


def create_body(
    parent_element,
    name,
    pos,
    geom_type="capsule",
    geom_len=0.10,
    geom_size=0.05,
    geom_dir=(0, 0, -1),
    geom_pos="origin",
    joint_attrs=None,
    joint_names=("joint",),
    joint_axes=((1, 0, 0),),
    joint_poses=None,
    **kwargs,
):
    """Create a body with joints and a geometry."""
    body = ET_sub(parent_element, "body", {"name": name, "pos": pos})
    ET_set(body, kwargs)
    for index, (joint_name, joint_axis) in enumerate(zip(joint_names, joint_axes)):
        joint_kwargs = {}
        if joint_poses is not None and index < len(joint_poses):
            joint_kwargs["pos"] = joint_poses[index]
        create_joint(
            body,
            joint_name,
            axis=joint_axis,
            joint_attrs=joint_attrs,
            **joint_kwargs,
        )

    geom_size = default_rng(geom_size, 0.05)
    geom_len = default_rng(geom_len, 0.10)
    geom_dir = numpy.asarray(geom_dir, dtype=float)
    direction_norm = numpy.linalg.norm(geom_dir)
    if direction_norm == 0:
        raise ValueError("geom_dir must be non-zero")
    geom_ft = geom_dir / direction_norm * geom_len

    if geom_pos == "mirror":
        fromto = (-0.5 * geom_ft).tolist() + (0.5 * geom_ft).tolist()
        pos = fromto[0:3]
    elif geom_pos == "origin":
        fromto = [0, 0, 0] + geom_ft.tolist()
        pos = [0, 0, 0]
    elif isinstance(geom_pos, (list, tuple)):
        if len(geom_pos) == 3:
            pos = geom_pos
            fromto = list(pos) + (geom_ft + numpy.asarray(pos)).tolist()
        elif len(geom_pos) == 6:
            fromto = geom_pos
            pos = fromto[0:3]
        else:
            raise ValueError("Invalid geom_pos length")
    else:
        raise ValueError(f"Invalid geom_pos: {geom_pos}")

    if geom_type in ("capsule", "cylinder", "box", "ellipsoid"):
        ET_sub(
            body,
            "geom",
            {
                "name": f"{name}1",
                "type": geom_type,
                "fromto": fromto,
                "size": geom_size,
            },
        )
    elif geom_type == "sphere":
        ET_sub(
            body,
            "geom",
            {"name": f"{name}1", "type": geom_type, "pos": pos, "size": geom_size},
        )
    else:
        raise ValueError(f"Unsupported geom_type: {geom_type}")
    return body


def create_random_humanoid(noise_scale=1.0):
    """Create a valid, physically bounded random Humanoid XML string."""
    noise_scale = _validate_noise_scale(noise_scale)
    root = ET.Element("mujoco", {"model": "random_humanoid"})
    prepare_assets(root)
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "light",
        {
            "cutoff": "100",
            "diffuse": "1 1 1",
            "dir": "0 0 -1.3",
            "directional": "true",
            "exponent": "1",
            "pos": "0 0 1.3",
            "specular": ".1 .1 .1",
        },
    )
    floor_friction = _sample_relative(1.0, 0.30, noise_scale)
    ET.SubElement(
        worldbody,
        "geom",
        {
            "condim": "3",
            "friction": f"{floor_friction:.6f} .1 .1",
            "type": "plane",
            "size": "20 20 0.125",
            "material": "MatPlane",
            "name": "floor",
            "pos": "0 0 0",
            "rgba": "0.8 0.9 0.8 1",
        },
    )

    body_attrs = sample_all_limb_sizes(noise_scale=noise_scale)
    joint_attrs = sample_all_joint_attributes(
        asymmetric=False, noise_scale=noise_scale
    )
    torso = ET_sub(
        worldbody,
        "body",
        {"name": "torso", "pos": (0, 0, body_attrs["torso_height"])},
    )
    ET_sub(
        torso,
        "camera",
        {
            "name": "track",
            "mode": "trackcom",
            "pos": "0 -4 0",
            "xyaxes": "1 0 0 0 0 1",
        },
    )
    create_joint(
        torso,
        "root",
        armature=0.0,
        damping=0.0,
        stiffness=0.0,
        type="free",
    )
    ET_sub(
        torso,
        "geom",
        {
            "name": "torso1",
            "fromto": (
                0,
                -body_attrs["len_torso1"] / 2,
                0,
                0,
                body_attrs["len_torso1"] / 2,
                0,
            ),
            "size": body_attrs["size_torso1"],
            "type": "capsule",
        },
    )
    ET_sub(
        torso,
        "geom",
        {
            "name": "head",
            "pos": (0, 0, body_attrs["t2h"]),
            "size": body_attrs["size_head"],
            "type": "sphere",
            "user": "258",
        },
    )
    ET_sub(
        torso,
        "geom",
        {
            "name": "uwaist",
            "fromto": (
                body_attrs["dx_uwaist"],
                -body_attrs["len_uwaist"] / 2,
                -body_attrs["u2t"],
                body_attrs["dx_uwaist"],
                body_attrs["len_uwaist"] / 2,
                -body_attrs["u2t"],
            ),
            "size": body_attrs["size_uwaist"],
            "type": "capsule",
        },
    )

    lwaist = create_body(
        torso,
        "lwaist",
        pos=[body_attrs["dx_lwaist"], 0, -body_attrs["l2t"]],
        geom_len=body_attrs["len_lwaist"],
        geom_size=body_attrs["size_lwaist"],
        geom_pos="mirror",
        geom_dir=(0, 1, 0),
        joint_names=("abdomen_z", "abdomen_y"),
        joint_attrs=joint_attrs,
        joint_axes=((0, 0, 1), (0, 1, 0)),
        joint_poses=((0, 0, body_attrs["lwaist_joint_pos"]),) * 2,
    )
    pelvis = create_body(
        lwaist,
        "pelvis",
        pos=[0, 0, -body_attrs["p2l"]],
        geom_len=body_attrs["len_pelvis"],
        geom_size=body_attrs["size_pelvis"],
        geom_dir=(0, 1, 0),
        geom_pos="mirror",
        joint_axes=((1, 0, 0),),
        joint_poses=((0, 0, body_attrs["pelvis_lwaist_joint_pos"]),),
        joint_attrs=joint_attrs,
        joint_names=("abdomen_x",),
    )

    for side in ("right", "left"):
        sign = 1 if side == "left" else -1
        thigh_direction = numpy.asarray(
            (0, sign * body_attrs["lthigh_direction"][1], -1.0), dtype=float
        )
        thigh_direction /= numpy.linalg.norm(thigh_direction)
        thigh = create_body(
            pelvis,
            f"{side}_thigh",
            pos=[
                body_attrs["th2p_x"],
                sign * body_attrs["th2p_y"],
                -body_attrs["th2p_z"],
            ],
            geom_len=body_attrs["len_thigh"],
            geom_size=body_attrs["size_thigh"],
            geom_dir=thigh_direction,
            joint_names=(
                f"{side}_hip_x",
                f"{side}_hip_z",
                f"{side}_hip_y",
            ),
            joint_attrs=joint_attrs,
            joint_axes=((-sign, 0, 0), (0, 0, -sign), (0, 1, 0)),
        )
        shin = create_body(
            thigh,
            f"{side}_shin",
            pos=thigh_direction * body_attrs["s2th"],
            geom_len=body_attrs["len_shin"],
            geom_size=body_attrs["size_shin"],
            geom_dir=(0, 0, -1),
            joint_attrs=joint_attrs,
            joint_names=(f"{side}_knee",),
            joint_axes=((0, -1, 0),),
            joint_poses=((0, 0, body_attrs["knee_joint_pos"]),),
        )
        foot_geom_offset = body_attrs["size_foot"] * (4.0 / 3.0)
        foot = ET_sub(
            shin,
            "body",
            {
                "name": f"{side}_foot",
                "pos": (0, 0, -(body_attrs["f2s"] + foot_geom_offset)),
            },
        )
        ET_sub(
            foot,
            "geom",
            {
                "name": f"{side}_foot",
                "pos": (0, 0, foot_geom_offset),
                "size": body_attrs["size_foot"],
                "type": "sphere",
                "user": "0",
            },
        )

    for side in ("right", "left"):
        sign = 1 if side == "left" else -1
        upper_direction = numpy.asarray((1, sign, -1), dtype=float)
        upper_direction /= numpy.linalg.norm(upper_direction)
        lower_direction = numpy.asarray((1, -sign, 1), dtype=float)
        lower_direction /= numpy.linalg.norm(lower_direction)
        upper_arm = create_body(
            torso,
            f"{side}_upper_arm",
            pos=[0, sign * body_attrs["ua2t_y"], body_attrs["ua2t"]],
            geom_len=body_attrs["len_upper_arm"],
            geom_size=body_attrs["size_upper_arm"],
            geom_dir=upper_direction,
            joint_names=(f"{side}_shoulder1", f"{side}_shoulder2"),
            joint_attrs=joint_attrs,
            joint_axes=((2, -sign, 1), (0, sign, 1)),
        )
        lower_arm = create_body(
            upper_arm,
            f"{side}_lower_arm",
            pos=upper_direction * body_attrs["la2ua"],
            geom_len=body_attrs["len_lower_arm"],
            geom_size=body_attrs["size_lower_arm"],
            geom_dir=lower_direction,
            geom_pos=(
                body_attrs["la_origin"],
                -body_attrs["la_origin"] * sign,
                body_attrs["la_origin"],
            ),
            joint_names=(f"{side}_elbow",),
            joint_attrs=joint_attrs,
            joint_axes=((0, -1, -sign),),
        )
        ET_sub(
            lower_arm,
            "geom",
            {
                "name": f"{side}_hand",
                "pos": lower_direction * body_attrs["h2la"],
                "size": body_attrs["size_hand"],
                "type": "sphere",
                "user": "0",
            },
        )

    tendon = ET.SubElement(root, "tendon")
    tendon_factor = _sample_relative(1.0, 0.20, noise_scale)
    for side in ("left", "right"):
        hipknee = ET_sub(tendon, "fixed", {"name": f"{side}_hipknee"})
        ET_sub(
            hipknee,
            "joint",
            {"coef": -tendon_factor, "joint": f"{side}_hip_y"},
        )
        ET_sub(
            hipknee,
            "joint",
            {"coef": tendon_factor, "joint": f"{side}_knee"},
        )

    actuator = ET.SubElement(root, "actuator")
    actuator_order = (
        "abdomen_y",
        "abdomen_z",
        "abdomen_x",
        "right_hip_x",
        "right_hip_z",
        "right_hip_y",
        "right_knee",
        "left_hip_x",
        "left_hip_z",
        "left_hip_y",
        "left_knee",
        "right_shoulder1",
        "right_shoulder2",
        "right_elbow",
        "left_shoulder1",
        "left_shoulder2",
        "left_elbow",
    )
    sample_joints_gear(actuator, actuator_order, noise_scale=noise_scale)
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")


def humanoid_xml_sampler(output_file, noise_scale=1.0):
    xml_str = create_random_humanoid(noise_scale=noise_scale)
    with open(output_file, "w", encoding="utf-8") as file:
        file.write(xml_str)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a random humanoid model using MuJoCo."
    )
    parser.add_argument(
        "--output", "-o", default="random_humanoid.xml", help="Output XML file path"
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Randomization strength in [0, 1]",
    )
    args = parser.parse_args()
    humanoid_xml_sampler(args.output, noise_scale=args.noise_scale)

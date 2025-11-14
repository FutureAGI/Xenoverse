import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
import numpy

def default_rng(range_tuple, default=0):
    if(isinstance(range_tuple, (int, float))):
        return range_tuple
    elif(isinstance(range_tuple, (list, tuple)) and len(range_tuple) == 2):
        return random.uniform(*range_tuple)
    else:
        return default

def sample_joint_attributes(armature_range, damping_range, stiffness_range, limit_lower, limit_higher=None, joint_type="hinge"):
    """Sample joint attributes within given ranges"""
    armature = random.uniform(*armature_range)
    damping = random.uniform(*damping_range)
    stiffness = random.uniform(*stiffness_range)

    range_lower = random.uniform(limit_lower[0], limit_lower[1])
    if(limit_higher is None):
        range_higher = -range_lower
    else:
        range_higher = random.uniform(limit_higher[0], limit_higher[1])

    if(range_lower > range_higher):
        range_lower, range_higher = range_higher, range_lower
    return {"armature": armature, 
            "damping": damping, 
            "stiffness": stiffness, 
            "type": joint_type,
            "range": (range_lower, range_higher)}

def sample_joints_gear(parent, joint_names):
    """Sample gear values for a list of joint names"""
    joints_gear = {}
    joints_gear["abdomen_x"] = random.uniform(50, 200)
    joints_gear["abdomen_y"] = joints_gear["abdomen_x"]
    joints_gear["abdomen_z"] = joints_gear["abdomen_x"]
    joints_gear["knee"] = random.uniform(80, 400)
    joints_gear["hip_x"] = random.uniform(50, 200) 
    joints_gear["hip_z"] = random.uniform(50, 200) 
    joints_gear["hip_y"] = random.uniform(150, 500)
    joints_gear["shoulder1"] = random.uniform(20, 50)
    joints_gear["shoulder2"] = random.uniform(20, 50)
    joints_gear["elbow"] = random.uniform(20, 50)

    for joint_name in joint_names:
        joint_key = joint_name.replace("left_", "").replace("right_", "")
        ET_sub(parent, "motor", {
            "joint": joint_name,
            "gear": joints_gear[joint_key],
            "name": joint_name
        })
    return

def perturb(kw, sparsity=0.10, scale=0.33, asymmetric=True):
    new_kw = kw.copy()
    if(asymmetric):
        return new_kw
    for k,v in kw.items():
        if(isinstance(v, (int, float))):
            sc = random.uniform(max(0.3, 1-scale), 1+scale)
            if(random.random() < sparsity):
                new_kw[k] = v * sc
        elif(isinstance(v, (list, tuple))):
            v_new = list(v)
            for i in range(len(v)):
                sc = random.uniform(max(0.3, 1-scale), 1+scale)
                if(random.random() < sparsity):
                    v_new[i] = v[i] * sc
            new_kw[k] = type(v)(v_new)
        else:
            new_kw[k] = v
    return new_kw

def strval(x):
    if(isinstance(x, float)):
        return str(round(x, 6))
    else:
        return str(x)
        
def ET_sub(parent_element, tag, attrib={}, text=None):
    """Helper function to create a subelement with text and attributes"""
    new_attrib = {}
    for k, v in attrib.items():
        if(v is None):
            new_attrib[k] = ""
        elif(isinstance(v, (list, tuple))):
            new_attrib[k] = " ".join(map(strval, v))
        else:
            new_attrib[k] = strval(v)
    element = ET.SubElement(parent_element, tag, new_attrib)
    if text is not None:
        element.text = text
    return element

def ET_set(element, attrib={}):
    """Helper function to set attributes of an element"""
    for k, v in attrib.items():
        if(v is None):
            element.set(k, "")
        elif(isinstance(v, (list, tuple))):
            element.set(k, " ".join(map(strval, v)))
        else:
            element.set(k, strval(v))
    return element

def sample_all_joint_attributes(asymmetric=True, noise_scale=1.0):
    """Sample all joint attributes for a humanoid"""
    ub = 1.0 + noise_scale
    lb = 1.0 / ub
    dr = (5 * lb, 5 * ub)
    sr1 = (10 * lb, 10 * ub)
    sr2 = (20 * lb, 20 * ub)
    sr3 = (lb, ub)
    ar1 = (0.02 * lb, 0.02 * ub)
    ar2 = (0.01 * lb, 0.01 * ub)
    ar3 = (0.005 * lb, 0.005 * ub)
    ar4 = (0.003 * lb, 0.003 * ub)

    joint_attrs = {}
    joint_attrs["abdomen_z"] = sample_joint_attributes(ar1, dr, sr2, (-90, -30), (30, 90))
    joint_attrs["abdomen_y"] = sample_joint_attributes(ar1, dr, sr1, (-120, -45), (15, 60))
    joint_attrs["abdomen_x"] = sample_joint_attributes(ar1, dr, sr1, (-75, -15))

    joint_attrs["left_hip_x"] = sample_joint_attributes(ar2, dr, sr1, (-50, -15), (0, 15))
    joint_attrs["left_hip_y"] = sample_joint_attributes(ar2, dr, sr2, (-90, -30), (15, 70))
    joint_attrs["left_hip_z"] = sample_joint_attributes(ar2, dr, sr1, (-160, -80), (10, 40))

    joint_attrs["left_knee"] = sample_joint_attributes(ar2, dr, sr2, (-160, -90), (-20, 5))

    joint_attrs["left_shoulder1"] = sample_joint_attributes(ar3, dr, sr3, (-120, -30), (30, 120))
    joint_attrs["left_shoulder2"] = sample_joint_attributes(ar3, dr, sr3, (-120, -30), (30, 120))
    joint_attrs["left_elbow"] = sample_joint_attributes(ar4, dr, sr3, (-160, -45), (30, 90))

    for ljname in ["left_hip_x", "left_hip_y", "left_hip_z", "left_knee", "left_shoulder1", "left_shoulder2", "left_elbow"]:
        rjname = ljname.replace("left", "right")
        joint_attrs[rjname] = perturb(joint_attrs[ljname], asymmetric=asymmetric)
    
    return joint_attrs


"""
Sample all limb sizes
"""
def sample_all_limb_sizes(noise_scale=1.0):
    """Sample all limb sizes for a humanoid"""
    sizes = {}
    ub = 1.0 + noise_scale
    lb = 1.0 / ub
    sizes["size_head"] = 0.09 * random.uniform(lb, ub)
    sizes["size_torso1"] = 0.07 * random.uniform(lb, ub)
    sizes["len_torso1"] = 0.14 * random.uniform(lb, ub)
    sizes["size_uwaist"] = 0.06 * random.uniform(lb, ub) 
    sizes["len_uwaist"] = 0.12 * random.uniform(lb, ub) 
    sizes["dx_uwaist"] = random.uniform(-0.03, 0.03)
    sizes["size_lwaist"] = 0.06 * random.uniform(lb, ub)  
    sizes["dx_lwaist"] = random.uniform(-0.03, 0.03)
    sizes["len_lwaist"] = 0.12 * random.uniform(lb, ub) 
    sizes["size_pelvis"] = 0.09 * random.uniform(lb, ub) 
    sizes["len_pelvis"] = 0.14 * random.uniform(lb, ub) 
    sizes["size_thigh"] = 0.06 * random.uniform(lb, ub) 
    sizes["len_thigh"] = 0.35 * random.uniform(lb, ub) 
    sizes["size_shin"] = 0.049 * random.uniform(lb, ub) 
    sizes["len_shin"] = 0.30 * random.uniform(lb, ub) 
    sizes["size_foot"] = 0.075 * random.uniform(lb, ub) 
    sizes["size_upper_arm"] = 0.04 * random.uniform(lb, ub) 
    sizes["len_upper_arm"] = 0.16 * random.uniform(lb, ub) 
    sizes["size_lower_arm"] = 0.03 * random.uniform(lb, ub) 
    sizes["len_lower_arm"] = 0.16 * random.uniform(lb, ub) 
    sizes["size_hand"] = 0.04 * random.uniform(lb, ub) 

    # torso1 to head
    sizes["t2h"] = sizes["size_torso1"] + sizes["size_head"] + random.uniform(-0.01, 0.02)
    # upper waist1 to torso1
    sizes["u2t"] = sizes["size_uwaist"] + sizes["size_torso1"] + random.uniform(-0.01, 0.02)
    # lower waist1 to torso1
    sizes["l2t"] = 2 * sizes["size_uwaist"] + sizes["size_torso1"] + sizes["size_lwaist"] + random.uniform(-0.01, 0.02)
    # pelvis to lower waist1
    sizes["p2l"] = sizes["size_pelvis"] + sizes["size_lwaist"] + random.uniform(-0.01, 0.02)
    # thigh root to pelvis
    sizes["th2p_x"] = random.uniform(-0.02, 0.02)
    sizes["th2p_y"] = sizes["len_pelvis"] / 2 + random.uniform(-0.01, 0.01)
    sizes["th2p_z"] = sizes["size_pelvis"] * random.uniform(-0.05, 0.05)
    # shin to thigh
    sizes["s2th"] = sizes["len_thigh"] + sizes["size_thigh"] + sizes["size_shin"] + random.uniform(-0.01, 0.02)
    # foot to shin
    sizes["f2s"] = sizes["len_shin"] + sizes["size_shin"] + sizes["size_foot"] + random.uniform(-0.01, 0.02)
    sizes["foot_height"] = sizes["size_foot"] + random.uniform(-0.01, 0.02)
    # upper arm to torso1
    sizes["ua2t"] = sizes["size_torso1"] + random.uniform(-0.01, 0.02)
    sizes["ua2t_y"] = sizes["len_torso1"] / 2 + sizes["size_upper_arm"] + random.uniform(-0.01, 0.02)
    # lower arm to upper arm
    sizes["la2ua"] = sizes["len_upper_arm"] + sizes["size_lower_arm"] + random.uniform(-0.01, 0.02)
    # hand to lower arm
    sizes["h2la"] = sizes["len_lower_arm"] + sizes["size_hand"] + random.uniform(-0.01, 0.02)
    sizes["torso_height"] = (sizes["t2h"] + sizes["u2t"] + sizes["l2t"] + sizes["p2l"] +
                                sizes["th2p_z"] + sizes["s2th"] + sizes["f2s"] + sizes["size_foot"])
    # lower arm origin offset
    sizes["la_origin"] = random.uniform(-0.01, 0.02)
    # pelvis to lower waist joint position
    sizes["pelvis_lwaist_joint_pos"] = sizes["size_pelvis"] + random.uniform(-0.01, 0.02)

    # thigh direction
    sizes["lthigh_direction"] = (0, -random.uniform(0.0, 0.1), -1.0)

    return sizes

def prepare_assets(root):
    ET.SubElement(root, "compiler", {"angle": "degree", "inertiafromgeom": "true"})
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", {"armature":"1", "damping":"1", "limited":"true"})
    ET.SubElement(default, "geom", {"conaffinity":"1", "condim":"1", "contype":"1", "margin":"0.001", "material":"geom", "rgba":"0.8 0.6 .4 1"})
    ET.SubElement(default, "motor", {"ctrllimited":"true", "ctrlrange":"-.4 .4"})

    ET.SubElement(root, "option", {"integrator":"RK4", "iterations":"50", "solver":"PGS", "timestep":"0.003"})
    size = ET.SubElement(root, "size", {"nkey":"5", "nuser_geom":"1"})
    visual = ET.SubElement(root, "visual")
    map = ET.SubElement(visual, "map", {"fogend":"5", "fogstart":"3"})
    ET.SubElement(root, "asset")
    asset = root.find("./asset")
    ET.SubElement(asset, "texture", {"builtin":"gradient", "height":"100", "rgb1":".4 .5 .6", "rgb2":"0 0 0", "type":"skybox", "width":"100"})
    ET.SubElement(asset, "texture", {"builtin":"flat", "height":"1278", "mark":"cross", "markrgb":"1 1 1", "name":"texgeom", "random":"0.01", "rgb1":"0.8 0.6 0.4", "rgb2":"0.8 0.6 0.4", "type":"cube", "width":"127"})
    ET.SubElement(asset, "texture", {"builtin":"checker", "height":"100", "name":"texplane", "rgb1":"0 0 0", "rgb2":"0.8 0.8 0.8", "type":"2d", "width":"100"})
    ET.SubElement(asset, "material", {"name":"MatPlane", "reflectance":"0.5", "shininess":"1", "specular":"1", "texrepeat":"60 60", "texture":"texplane"})
    ET.SubElement(asset, "material", {"name":"geom", "texture":"texgeom", "texuniform":"true"})

def create_joint(parent_element, name, 
                axis=(0,1,0), 
                type="hinge",
                joint_attrs={},
                **kwargs):
    """Create a joint element and attach it to the parent element"""
    joint = ET_sub(parent_element, "joint", 
            attrib={"name": name,
                    "type": type,
                    "axis": axis})
    if(name in joint_attrs):
        kwargs.update(joint_attrs[name])
    joint = ET_set(joint, kwargs)
    return joint
 
def create_body(parent_element, name, pos,
            geom_type="capsule",
            geom_len=0.10,
            geom_size=0.05,
            geom_dir=(0, 0, -1),
            geom_pos="origin", # "origin", "mirror", tuple
            joint_attrs={},
            joint_names=["joint"],
            joint_axes=[(1, 0, 0)],
            joint_poses=None,
            **kwargs):
    """Create a body element with a joint and geometry"""
    body = ET_sub(parent_element, "body", {
        "name": name,
        "pos": pos,
    })
    body = ET_set(body, kwargs)

    # Create a joint based on the body name
    ret_joint_names = []
    for i, (joint_name, joint_axis) in enumerate(
            zip(joint_names, joint_axes)):
        if(joint_poses is not None and i < len(joint_poses)):
            create_joint(body, joint_name, 
                                axis=joint_axis, 
                                pos=joint_poses[i],
                                joint_attrs=joint_attrs)
        else:
            create_joint(body, joint_name, 
                                axis=joint_axis, 
                                joint_attrs=joint_attrs)

    geom_size = default_rng(geom_size, 0.05)
    geom_len = default_rng(geom_len, 0.10)
    geom_dir = numpy.array(geom_dir)
    geom_dir = geom_dir / numpy.linalg.norm(geom_dir)
    geom_ft = geom_dir * geom_len

    if(geom_pos == "mirror"):
        fromto = (-0.5 * geom_ft).tolist() + (0.5 * geom_ft).tolist()
        pos = fromto[0:3]
    elif(geom_pos == "origin"):
        fromto = [0, 0, 0] + geom_ft.tolist()
        pos = [0, 0, 0]
    elif(isinstance(geom_pos, (list, tuple))):
        if(len(geom_pos) == 3):
            pos = geom_pos
            fromto = list(pos) + (geom_ft + numpy.array(pos)).tolist()
        elif(len(geom_pos) == 6):
            fromto = geom_pos
        else:
            raise ValueError("Invalid geom_pos length")
    else:
        raise ValueError(f"Invalid geom_pos: {geom_pos}")

    # Create a geometry element
    if(geom_type in ["capsule", "cylinder", "box", "ellipsoid"]):
        ET_sub(body, "geom", {
            "name": f"{name}1",
            "type": geom_type,
            "fromto": fromto,
            "size": geom_size
        })
    elif(geom_type == "sphere"):
        ET_sub(body, "geom", {
            "name": f"{name}1",
            "type": geom_type,
            "pos": pos,
            "size": geom_size
        })
    else:
        raise ValueError(f"Unsupported geom_type: {geom_type}")
    return body
 
def create_random_humanoid(noise_scale=1.0):
    # Create the xml root element
    root = ET.Element("mujoco", {"model": "random_humanoid"})
    prepare_assets(root)
    
    # Create the worldbody element
    worldbody = ET.SubElement(root, "worldbody")

    frac_base = random.uniform(0.2, 1.6)
    # Create environments
    ET.SubElement(worldbody, "geom", {
        "condim": "3",
        "friction": f"{frac_base} .1 .1",
        "type": "plane",
        "size": "20 20 0.125",
        "material": "MatPlane",
        "name": "floor",
        "pos": "0 0 0",
        "rgba": "0.8 0.9 0.8 1"
    })
    
    body_attrs = sample_all_limb_sizes(noise_scale=noise_scale)
    joints_attrs = sample_all_joint_attributes(asymmetric=True, noise_scale=noise_scale)

    # Create the root body with a free joint
    torso = ET_sub(worldbody, "body", {"name": "torso", "pos": (0, 0, body_attrs["torso_height"])})
    create_joint(torso, "root", armature=0.0, damping=0.0, stiffness=0.0,  type="free")

    # Create the torso1, head, waist, pelvis
    torso1 = ET_sub(torso, "geom", 
                        {"name": "torso1", 
                        "pos": (0, 0, 0),
                        "fromto": (0, -body_attrs["len_torso1"] / 2, 0, 0, body_attrs["len_torso1"] / 2, 0),
                        "size": body_attrs["size_torso1"],
                        "type": "capsule"})

    head = ET_sub(torso, "geom", 
                        {"name": "head", 
                        "pos": (0, 0, body_attrs["t2h"]),
                        "size": body_attrs["size_head"],
                        "type": "sphere"})

    uwaist = ET_sub(torso, "geom", 
                        {"name": "uwaist", 
                        "fromto": (body_attrs["dx_uwaist"], -body_attrs["len_uwaist"] / 2, -body_attrs["u2t"], 
                                    body_attrs["dx_uwaist"], body_attrs["len_uwaist"] / 2, -body_attrs["u2t"]),
                        "size": body_attrs["size_uwaist"],
                        "type": "capsule"})

    # Create lower waist with one hinge joint
    lwaist = create_body(torso, "lwaist", 
                pos=[body_attrs["dx_lwaist"], 0, -body_attrs["l2t"]],
                geom_type="capsule",
                geom_len=body_attrs["len_lwaist"],
                geom_size=body_attrs["size_lwaist"],
                geom_pos="mirror",
                geom_dir=(0, 1, 0),
                joint_names=["abdomen_z", "abdomen_y"],
                joint_attrs = joints_attrs,
                joint_axes=[(0, 0, 1), (0, 1, 0)])

    # Create pelvis with two hinge joints
    pelvis = create_body(lwaist, "pelvis", 
                pos=[0, 0, -body_attrs["p2l"]],
                geom_type="capsule",
                geom_len=body_attrs["len_pelvis"],
                geom_size=body_attrs["size_pelvis"],
                geom_dir=(0, 1, 0),
                geom_pos="mirror",
                joint_axes=[(1, 0, 0)],
                joint_poses=[(0, 0, body_attrs["pelvis_lwaist_joint_pos"])],
                joint_attrs = joints_attrs,
                joint_names=["abdomen_x"])

    # Create legs
    for side in ["left", "right"]:
        sign = 1 if side == "left" else -1

        joint_names = [f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"]
        thigh = create_body(pelvis, f"{side}_thigh", 
                    pos=[body_attrs["th2p_x"], sign * body_attrs["th2p_y"], -body_attrs["th2p_z"]],
                    geom_type="capsule",
                    geom_len=body_attrs["len_thigh"],
                    geom_size=body_attrs["size_thigh"],
                    geom_pos="origin",
                    geom_dir=(body_attrs["lthigh_direction"][0], sign * body_attrs["lthigh_direction"][1], body_attrs["lthigh_direction"][2]),
                    joint_names=[f"{side}_hip_x", f"{side}_hip_y", f"{side}_hip_z"],
                    joint_attrs = joints_attrs,
                    joint_axes=[(-sign, 0, 0), (0, 1, 0), (0, 0, -sign)],
                    )

        shin = create_body(thigh, f"{side}_shin", 
                    pos=[0, 0, -body_attrs["s2th"]],
                    geom_type="capsule",
                    geom_len=body_attrs["len_shin"],
                    geom_size=body_attrs["size_shin"],
                    geom_pos="origin",
                    geom_dir=(0, 0, -1),
                    joint_attrs = joints_attrs,
                    joint_names = [f"{side}_knee"],
                    joint_axes=[(1, 0, 0)])

        foot = ET_sub(shin, "geom", 
                        {"name": f"{side}_foot", 
                        "pos": (0, 0, -body_attrs["f2s"]),
                        "size": body_attrs["size_foot"],
                        "type": "sphere",
                        "user": "0"})
    
    # Create arms
    norm_factor = 1/(numpy.sqrt(3))

    for side in ["left", "right"]:
        sign = 1 if side == "left" else -1

        upper_arm = create_body(torso, f"{side}_upper_arm", 
                    pos=[0, sign * body_attrs["ua2t_y"], body_attrs["ua2t"]],
                    geom_type="capsule",
                    geom_len=body_attrs["len_upper_arm"],
                    geom_size=body_attrs["size_upper_arm"],
                    geom_dir=(1, sign, -1),
                    geom_pos="origin",
                    joint_names = [f"{side}_shoulder1", f"{side}_shoulder2"],
                    joint_attrs = joints_attrs,
                    joint_axes=[(2, -sign, 1), (0, sign, 1)]
                    )

        
        lower_arm = create_body(upper_arm, f"{side}_lower_arm", 
                    pos=[norm_factor * body_attrs["la2ua"], norm_factor * body_attrs["la2ua"] * sign, -norm_factor * body_attrs["la2ua"]],
                    geom_type="capsule",
                    geom_len=body_attrs["len_lower_arm"],
                    geom_size=body_attrs["size_lower_arm"],
                    geom_dir=(1, -sign, 1),
                    geom_pos=(body_attrs["la_origin"], -body_attrs["la_origin"] * sign, body_attrs["la_origin"]),
                    joint_names = [f"{side}_elbow"],
                    joint_attrs = joints_attrs,
                    joint_axes=[(0, -1, -sign)],
                    )

        hand = ET_sub(lower_arm, "geom", 
                        {"name": f"{side}_hand", 
                        "pos": (norm_factor * body_attrs["h2la"], norm_factor * body_attrs["h2la"] * -sign, norm_factor * body_attrs["h2la"]),
                        "size": body_attrs["size_hand"],
                        "type": "sphere",
                        "user": "0"})
    
    # Add tendon elements for muscles
    tendon = ET.SubElement(root, "tendon")
    factor = random.uniform(0.0, 1.0)
    lhk = ET_sub(tendon, "fixed", {"name": "left_hipknee"})
    rhk = ET_sub(tendon, "fixed", {"name": "right_hipknee"})
    ET_sub(lhk, "joint", {"coef": -factor, "joint": "left_hip_y"})
    ET_sub(lhk, "joint", {"coef": factor, "joint": "left_knee"})
    ET_sub(rhk, "joint", {"coef": -factor, "joint": "right_hip_y"})
    ET_sub(rhk, "joint", {"coef": factor, "joint": "right_knee"})

    # Add actuator elements for muscles
    actuator = ET.SubElement(root, "actuator")
    sample_joints_gear(actuator, joints_attrs)
    
    # Generate the xml
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    return xml_str

def humanoid_xml_sampler(output_file, noise_scale=1.0):
    xml_str = create_random_humanoid(noise_scale=noise_scale)
    with open(output_file, 'w') as file:
        file.write(xml_str)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate a random humanoid model using MuJoCo.')
    parser.add_argument('--output', '-o', default='random_humanoid.xml', help='Output XML file path')
    args = parser.parse_args()
    humanoid_xml_sampler(args.output)
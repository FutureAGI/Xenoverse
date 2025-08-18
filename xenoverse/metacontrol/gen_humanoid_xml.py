import xml.etree.ElementTree as ET
from xml.dom import minidom
import random
 
def create_joint(parent_element, name, type="hinge", axis=(0,1,0), range=(-120, 120), **kwargs):
    """Create a joint element and attach it to the parent element"""
    joint = ET.SubElement(parent_element, "joint", {
        "name": name,
        "type": type,
        "axis": " ".join(map(str, axis)),
        "range": " ".join(map(str, range)),
        "armature": str(random.uniform(0.005, 0.02)),
        "damping": str(random.uniform(0.5, 3.0)),
        "stiffness": str(random.randint(5, 30))
    })
    for k, v in kwargs.items():
        joint.set(k, str(v))
    return joint
 
def create_body(parent_element, name, pos, mirror=False):
    """Create a body element with a joint and geometry"""
    body = ET.SubElement(parent_element, "body", {
        "name": name,
        "pos": " ".join(map(str, pos))
    })
    
    # Create a joint based on the body name
    if "shoulder" in name:
        create_joint(body, f"{name}_joint", type="ball", range=(-120, 120))
    elif "hip" in name:
        axis = (1,0,0) if mirror else (-1,0,0)
        create_joint(body, f"{name}_joint", type="hinge", axis=axis)
    else:
        create_joint(body, f"{name}_joint", type="hinge", range=(-150, 0))
    
    # Create a geometry element
    geom_type = random.choice(["capsule", "box", "ellipsoid"])
    ET.SubElement(body, "geom", {
        "type": geom_type,
        "size": " ".join([str(random.uniform(0.03, 0.08))] * (3 if geom_type == "box" else 1)),
        "rgba": f"{random.random()} {random.random()} {random.random()} 1"
    })
    return body
 
def create_asymmetric_humanoid():
    # Create the xml root element
    root = ET.Element("mujoco", {"model": "asymmetric_humanoid"})
    ET.SubElement(root, "compiler", {"angle": "degree", "inertiafromgeom": "true"})
    
    # Create the worldbody element
    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(worldbody, "geom", {
        "type": "plane",
        "size": "10 10 0.1",
        "rgba": "0.3 0.5 0.7 1"
    })
    
    # Create the root body with a free joint
    torso = ET.SubElement(worldbody, "body", {"name": "torso", "pos": "0 0 1.2"})
    create_joint(torso, "root", type="free")  # 正确传递父元素
    
    # Create the limbs
    for side, sign in [("left", 1), ("right", -1)]:
        # Create the thigh and shin
        thigh = create_body(torso, f"{side}_thigh", [0, sign*0.15, -0.2], mirror=(side=="right"))
        shin = create_body(thigh, f"{side}_shin", [0, 0, -0.4])
        
        # Create the arms 
        upper_arm = create_body(torso, f"{side}_upper_arm", [0, sign*0.2, 0.1])
        lower_arm = create_body(upper_arm, f"{side}_lower_arm", [0.3, 0, 0])
    
    # Generate the xml
    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    return xml_str

if __name__ == "__main__":
    xml_str = create_asymmetric_humanoid()
    print(xml_str)
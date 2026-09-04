import os
import xml.etree.ElementTree as ET

from gymnasium import error

try:
    from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
except error.DependencyNotInstalled as exc:
    _MUJOCO_IMPORT_ERROR = exc

    class HumanoidEnv:
        """Import placeholder that preserves XML sampling without MuJoCo."""

        def __init__(self, *args, **kwargs):
            raise error.DependencyNotInstalled(
                'MuJoCo is required for RandomHumanoidEnv; install "gymnasium[mujoco]"'
            ) from _MUJOCO_IMPORT_ERROR

from xenoverse.metacontrol.humanoid_xml_sampler import humanoid_xml_sampler
from xenoverse.utils import generate_secure_strings, pseudo_random_seed


MIN_HEALTHY_HEIGHT_RATIO = 0.70
MAX_HEALTHY_HEIGHT_RATIO = 1.40


def sample_humanoid(root_path=None, noise_scale=1.0):
    """Generate a random Humanoid task XML and return its absolute path.

    ``noise_scale`` is constrained to ``[0, 1]``. Zero produces the neutral
    Gymnasium-like morphology and one enables the full, bounded variation.
    """
    if root_path is None:
        root_path = os.path.dirname(os.path.abspath(__file__))
    root_path = os.path.abspath(os.path.join(root_path, "assets"))
    os.makedirs(root_path, exist_ok=True)
    file_id = generate_secure_strings(1, length=8)[0]
    file_path = os.path.join(root_path, f"random_humanoid_{file_id}.xml")
    humanoid_xml_sampler(file_path, noise_scale=noise_scale)
    return file_path


def get_humanoid_tasks(directory):
    # Acquire a list of tasks from the specified directory
    xml_files = [f for f in os.listdir(directory) if f.endswith(".xml")]
    xml_lists = []
    for xml_file in xml_files:
        if "random_humanoid" in xml_file:
            xml_lists.append(os.path.join(directory, xml_file))
    if not xml_lists:
        raise ValueError(f"No random_humanoid XML files found in directory: {directory}")
    return sorted(xml_lists)


class RandomHumanoidEnv(HumanoidEnv):
    """
    Humanoid environment whose MuJoCo model can be replaced with a task XML.
    """

    def __init__(self, seed=None, **kwargs):
        self.kwargs = kwargs
        super().__init__(**self.kwargs)
        self.seed(seed)

    def seed(self, seed=None):
        if seed is None:
            pseudo_random_seed(0)
        else:
            pseudo_random_seed(seed)

    def set_task(self, task):
        """Load a sampled XML and adapt termination to its standing height."""
        tree = ET.parse(task)
        root = tree.getroot()
        torso = root.find(".//body[@name='torso']")
        if torso is None:
            raise ValueError(f"Humanoid task has no torso body: {task}")
        torso_pos = torso.get("pos", "").split()
        if len(torso_pos) != 3:
            raise ValueError(f"Humanoid torso must have a 3D position: {task}")
        torso_height = float(torso_pos[2])
        if torso_height <= 0:
            raise ValueError(f"Humanoid torso height must be positive: {task}")

        init_kwargs = dict(self.kwargs)
        init_kwargs.pop("xml_file", None)
        init_kwargs.pop("healthy_z_range", None)
        if getattr(self, "mujoco_renderer", None) is not None:
            self.mujoco_renderer.close()
        super().__init__(
            xml_file=os.path.abspath(task),
            healthy_z_range=(
                torso_height * MIN_HEALTHY_HEIGHT_RATIO,
                torso_height * MAX_HEALTHY_HEIGHT_RATIO,
            ),
            **init_kwargs,
        )

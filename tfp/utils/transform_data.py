from scipy.signal import savgol_filter
import numpy as np
import os
import json
import tfp.config.config as config


def smoothen(joints, window=11, order=3, axis=0):
    return savgol_filter(joints, window, order, axis=axis)

class Transformation:

    def __init__(self,number_joints):
        """
        """
        self.parent_limbs = config.PARENT_LIMBS[number_joints]
        self.root = config.ROOT_LIMB[number_joints]
        self.head = config.HEAD_LIMB[number_joints]
        self.limb_ratios = config.LIMB_RATIOS[number_joints]


    def _gen_limb_graph(self):
        """
        returns a list whose each index i stores its corresponding children
        """
        n = len(parent_limbs)
        G = [[] for i in range(n)]
        for i in range(n):
            j = parent_limbs[i]
            if i != j:
                G[j].append(i)
        return G


    def _bfs_order(self):

        from collections import deque
        # Generate the graph
        G = _gen_limb_graph()
        # Store the bfs order of the joints
        q = deque([self.root])
        order = []
        while q:
            u = q.popleft()
            order.append(u)
            for v in G[u]:
                q.append(v)
        return order

    def _get_parent_relative_joint_locations(self,joints_xyz):
        """
        Params:
        joints_xyz: joint locations in different frames in Cartesian coordinates
        shape : [number_frames,number_joints,3]

        Returns:
        rel_joints_xyz: location of every joint relative to their respective parents
        shape : [number_frames,number_joints,3]
        """

        rel_joints_xyz = joints_xyz - joints_xyz[:,self.parent_limbs]

        return rel_joints_xyz

    def _get_abs_joint_locations(self,rel_joints_xyz):
        """
        Params:
        rel_joints_xyz: location of every joint relative to their respective parents
        shape : [number_frames,number_joints,3]

        Returns:
        abs_joints_xyz: location of every joint relative to their respective parents
        shape : [number_frames,number_joints,3]
        """


        abs_joints_xyz = np.zeros_like(rel_joints_xyz, dtype=np.float64)
        limb_order = _bfs_order()
        # Restore the initial coordinates of the root joint
        abs_joints_xyz[:,limb_order[0]] = self.abs_root_xyz
        # Traverse the bfs tree to calculate the joint coordinates
        for l in limb_order[1:]:
            p = parent_limbs[l]
            abs_joints_xyz[:,l] = abs_joints_xyz[:,p] + rel_joints_xyz[:,l]
        return abs_joints_xyz


    def _cart2sph(self,xyz):
        """
        Assumed shape: [number_frames,number_joints,3]
        """
        # Hypotenuse of the triangle formed in xy plane.It is required
        # to calculate the azimuth angle(φ)
        hxy = np.hypot(xyz[:,:,0],xyz[:,:,1])
        # Spherical coordinates matrix
        rtp = np.zeros_like(xyz, dtype=np.float64)
        # Radial distance(r)
        rtp[:,:,0] = np.linalg.norm(xyz, axis=2)
        # Polar angle(θ)
        rtp[:,:,1] = np.arctan2(hxy,xyz[:,:,2])
        # Azimuth angle(φ)
        rtp[:,:,2] = np.arctan2(xyz[:,:,1],xyz[:,:,0])
        return rtp


    def _sph2cart(self,rtp):
        """
        Assumed shape: [number_frames,number_joints,3]
        """
        # Cartesian coordinates matrix
        xyz = np.zeros_like(rtp, dtype=np.float64)
        # x coordinates: r * sin(θ) * cos(φ)
        xyz[:,:,0] = rtp[:,:,0] * np.sin(rtp[:,:,1]) * np.cos(rtp[:,:,2])
        # y coordinates: r * sin(θ) * sin(φ)
        xyz[:,:,1] = rtp[:,:,0] * np.sin(rtp[:,:,1]) * np.sin(rtp[:,:,2])
        # z coordinates: r * cos(θ)
        xyz[:,:,2] = rtp[:,:,0] * np.sin(rtp[:,:,1])
        return xyz


    def transform(self,joints, head_length=2.0):
        """
        Params:
        joints : the absolute joint coordinates of every frame before normalization
        shape : [number_frames,number_joints,3]

        Returns:
        cart_abs_joints : the absolute joint coordinates of every frame after normalization
        """
        # The absolute Cartesian coordinates of root joints of every frame
        self.abs_root_xyz = joints[:,self.root]
        # The Cartesian coordinates of all the joints relative to their respective parents
        rel_joints_xyz = self._get_parent_relative_joint_locations(joints)
        # The unnormalized joint coordinates in Spherical coordinate system
        sph_rel_joints = self. _cart2sph(rel_joints_xyz)
        # normalization of limb lengths
        fixed_limb_lengths = head_length * self.limb_ratios
        # Replace the limb lengths with unnormalized limb lengths
        sph_rel_joints[:, 0] = fixed_limb_lengths
        # Convert Spherical coordinates back to Cartesian coordinates
        cart_rel_joints = self._sph2cart(sph_rel_joints)
        # Recover the absolute coordinates
        cart_abs_joints = self._get_abs_joint_locations(cart_rel_joints)

        return cart_abs_joints

class GetData:
    r"""
        This class pick up the numpy files related to the input category
        and transform data using above functions, and after transformation store
        all file in th foder @<ROOT_of_REPO>/category
    """
    def __init__(self, data_location, category, num_joints = 21):
        self.data_loc = data_location
        self.category = category
        self.num_joints = num_joints
        self.label_file = config.LABEL_JSON_LOC

    def getdata(self):

        sav_dat_fol = os.path.join(os.getcwd(),self.category)
        transform = Transformation(self.num_joints)
        print("data transformation started.......")
        if not os.path.exists(sav_dat_fol):
            os.mkdir(sav_dat_fol)

        with open(self.label_file) as jsonfile:
            data_files = json.load(jsonfile)[self.category]

            for file_ in data_files:
                sub,trail = file_.split("_")
                data = np.load(os.path.join(self.data_loc,sub,file_+".npy"))

                _onlyjoint_data = []
                for frame in data:
                	_onlyjoint_data.append(frame[np.asarray(config.JOINT_INDEX[self.num_joints])])
                _data = transform.transform(np.asarray(_onlyjoint_data))
                # _data = _onlyjoint_data

                np.save(os.path.join(sav_dat_fol,file_+".npy"),_data)
        print("data transformation ended....")
        return None

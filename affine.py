# See here:
#   https://epubs.siam.org/doi/pdf/10.1137/16M1056936

import numpy as np
import scipy as sp
import scipy.linalg
np.set_printoptions(precision=4, suppress=True)


class LeeColors:

    @classmethod    
    def test_lee_colors(cls):
        print("should be 0", cls.euler_to_color(-np.pi))
        print("should be 255", cls.euler_to_color(np.pi))
        print("should be 127", cls.euler_to_color(0))
        print("should be -3.141", cls.color_to_euler(0))
        print("should be 3.141", cls.color_to_euler(255))
        print("should be 0", cls.color_to_euler(127))
        print("should be 0", cls.euler_to_color(cls.color_to_euler(0)))
        print("should be 255", cls.euler_to_color(cls.color_to_euler(255)))
        print("should be 127", cls.euler_to_color(cls.color_to_euler(127)))
        print("should be -3.141", cls.color_to_euler(cls.euler_to_color(-np.pi)))
        print("should be 3.141", cls.color_to_euler(cls.euler_to_color(np.pi)))
        print("should be 0", cls.color_to_euler(cls.euler_to_color(0)))
    
    @classmethod
    def euler_to_color(cls, e):
        # assuming euler is -pi .. pi
        v = (e + np.pi) / (2 * np.pi)
        return v * 255.0
        
    @classmethod
    def color_to_euler(cls, c):
        # assuming euler is -pi .. pi
        v = float(c / 255.0)
        e = (v * 2 * np.pi) - np.pi
        return e
    
    @classmethod
    def euler3_to_rgb(cls, euler3):
        rgb = [
            cls.euler_to_color(v)
            for v in euler3
        ]
        return rgb

    @classmethod
    def rgb_to_euler3(cls, rgb):
        euler3 = [
            cls.color_to_euler(v)
            for v in rgb
        ]
        return euler3

    @classmethod
    def scale_to_color(cls, scale):
        # assuming scale is -1 .. 1
        v = (scale + 1.0) / 2.0
        return v * 255.0

    @classmethod
    def color_to_scale(cls, color):
        # assuming scale is -1 .. 1
        v = color / 255.0
        s = (v * 2.0) - 1.0
        return s 

    @classmethod
    def scale3_to_rgb(cls, scale3):
        rgb = [
            cls.scale_to_color(v)
            for v in scale3
        ]
        return rgb

    @classmethod
    def rgb_to_scale3(cls, scale3):
        rgb = [
            cls.color_to_scale(v)
            for v in scale3
        ]
        return rgb
        
class LeeAffine:

    @classmethod
    def algebra_pos_matrix(cls, x, y, z):
        m = np.matrix(np.identity(4))
        m[0,3] = x
        m[1,3] = y
        m[2,3] = z
        return m

    @classmethod
    def algebra_rot_matrix(cls, x, y, z):
        m = np.matrix(np.zeros((3, 3)))
        m[1, 2] = x; m[2, 1] = -x
        m[0, 2] = y; m[2, 0] = -y
        m[0, 1] = z; m[1, 0] = -z  
        return m

    @classmethod
    def algebra_scale_matrix(cls, x, y, z):
        m = np.matrix(np.zeros((3, 3)))
        m[0, 0] = x
        m[1, 1] = y
        m[2, 2] = z
        return m

    @classmethod
    def m33_as_homogenous_4x4(cls, m3x3):
        m4x4 = np.matrix(np.zeros((4, 4)))
        for r in range(3):
            for c in range(3):
                m4x4[r, c] = m3x3[r, c]
        m4x4[3, 3] = 1
        return m4x4

    @classmethod
    def m44_as_linear_3x3(cls, m4x4):
        m3x3 = np.matrix(np.zeros((3, 3)))
        for r in range(3):
            for c in range(3):
                m3x3[r, c] = m4x4[r, c]
        return m3x3

    @classmethod
    def exp_map(cls, L_44, X_33, Y_33):
        return np.dot(
            L_44,
            cls.m33_as_homogenous_4x4(
                np.dot(
                    sp.linalg.expm(X_33),
                    sp.linalg.expm(Y_33)
                )
            )
        )

    @classmethod
    def log_map(cls, Aff_44):
        A_33 = cls.m44_as_linear_3x3(Aff_44)
        S_33 = sp.linalg.sqrtm(np.dot(A_33.T, A_33))
        L_44 = cls.algebra_pos_matrix(Aff_44[0, 3], Aff_44[1, 3], Aff_44[2, 3])
        X_33 = sp.linalg.logm(A_33 * sp.linalg.inv(S_33))
        Y_33 = sp.linalg.logm(S_33)
        return L_44, X_33, Y_33

    @classmethod
    def prs_to_affine(cls, pos, rot, scale):
        L_44 = cls.algebra_pos_matrix(pos[0], pos[1], pos[2])
        X_33 = cls.algebra_rot_matrix(rot[0], rot[1], rot[2])
        Y_33 = cls.algebra_scale_matrix(scale[0], scale[1], scale[2])
        Aff_44 = cls.exp_map(L_44, X_33, Y_33)
        return Aff_44
        
    @classmethod
    def affine_to_prs(cls, Aff_44):
        L_44, X_33, Y_33 = cls.log_map(Aff_44)
        pos = np.array([L_44[0,3], L_44[1,3], L_44[2,3]])
        rot = np.array([X_33[1,2], X_33[0,2], X_33[0,1]])
        scale = np.array([Y_33[0,0], Y_33[1,1], Y_33[2,2]])
        return pos, rot, scale
    
    @classmethod
    def algebra_to_affine(cls, alg):
        L_44 = cls.algebra_pos_matrix(alg[0], alg[1], alg[2])
        X_33 = cls.algebra_rot_matrix(alg[3], alg[4], alg[5])
        Y_33 = cls.algebra_scale_matrix(alg[6], alg[7], alg[8])
        Aff_44 = cls.exp_map(L_44, X_33, Y_33)
        return Aff_44

    @classmethod
    def affine_to_algebra(cls, Aff_44):
        pos, rot, scale = cls.affine_to_prs(Aff_44)
        return cls.prs_to_algebra(pos, rot, scale)

    @classmethod
    def prs_to_algebra(cls, pos, rot, scale):
        alg = np.array([
            pos[0], pos[1], pos[2],
            rot[0], rot[1], rot[2],
            scale[0], scale[1], scale[2]
        ])
        return alg
        
    @classmethod
    def algebra_to_prs(cls, alg):
        pos = np.array([alg[0], alg[1], alg[2]])
        rot = np.array([alg[3], alg[4], alg[5]])
        scale = np.array([alg[6], alg[7], alg[8]])
        return pos, rot, scale

    @classmethod
    def interpolate_algebra(cls, alg_a, alg_b, u):
        return alg_a + (alg_b - alg_a) * u
  

    @classmethod
    def basic_tests(cls):
        
        def test_fn(pos, rot, scale):
            alg = LeeAffine.prs_to_algebra(pos, rot, scale)
            aff = LeeAffine.algebra_to_affine(alg)
            n_pos, n_rot, n_scale = LeeAffine.affine_to_prs(aff)

            failed = False

            if not np.isclose(pos, n_pos).all():
                failed = True
                print("FAIL:", pos, "!=", n_pos)
                print("Diff", pos - n_pos)

            if not np.isclose(rot, n_rot).all():
                failed = True
                print("FAIL:", rot, "!=", n_rot)
                print("Diff", rot - n_rot)

            if not np.isclose(scale, n_scale).all():
                failed = True
                print("FAIL:", scale, "!=", n_scale)
                print("Diff", scale - n_scale)

            if not failed:
                print("PASSED for ", pos, rot, scale)
            
        test_fn(
            np.array([1, 0, 0]),
            np.array([0, 0, 0]),
            np.array([1, 1, 1])
        )

        test_fn(
            np.array([1, 2, 3]),
            np.array([1, 0, 0]),
            np.array([0.1, 0.3, 0.2])
        )


        for i in range(10):
            test_fn(
                np.random.random((3,)),
                np.random.random((3,)),
                np.random.random((3,))
            )

import numpy as np
import scipy.linalg as linalg
import scipy.sparse.linalg as LAs
import numpy.linalg as LA
import copy
import logging

import Sub180221 as Sub

pauli = np.array([
    [[1, 0],
     [0, 1]],
    [[0, 1],
     [1, 0]],
    [[0, -1j],
     [1j, 0]],
    [[1, 0],
     [0, -1.0]]
])

class iMPS: # arXiv:1810.07006
    def __init__(self, Ds, Dp):
        self.Dp = Dp
        self.Ds = Ds
        self.TL = np.random.rand(Ds, Dp, Ds) + 1j * np.random.rand(Ds, Dp, Ds)
        self.Canonical()

    def Transfer(self, direction=0):
        '''
        Calculate the Transfer matrix
        direction = 0: Left / 1: Right / 2: Both   
        '''
        if direction == 0 or direction == 2:
            self.EL = Sub.Group(Sub.NCon([self.TL, np.conj(self.TL)], [[-1, 1, -3], [-2, 1, -4]]), [[0, 1], [2, 3]])
        if direction == 1 or direction == 2:
            self.ER = Sub.Group(Sub.NCon([self.TR, np.conj(self.TR)], [[-1, 1, -3], [-2, 1, -4]]), [[0, 1], [2, 3]])

    def Transfer_two(self, other_MPS, direction=0):
        '''
        Calculate the Transfer matrix of <other_MPS|self>
        direction = 0: Left / 1: Right / 2: Both   
        '''
        EL = None
        ER = None
        if direction == 0 or direction == 2:
            EL = Sub.Group(Sub.NCon([self.TL, np.conj(other_MPS.TL)], [[-1, 1, -3], [-2, 1, -4]]), [[0, 1], [2, 3]])
        if direction == 1 or direction == 2:
            ER = Sub.Group(Sub.NCon([self.TR, np.conj(other_MPS.TR)], [[-1, 1, -3], [-2, 1, -4]]), [[0, 1], [2, 3]])
        return EL, ER

    def Canonical(self, prec=1e-12, step = 1000): # Algorithm 1 & 2
        # Promise to be left/right orthonormal, but may not be equivalent to T 

        self.Transfer()
        T = self.TL
        # We first try to solve the equation L * T * L^-1 = TL with the fixed point method
        eta, VL = LAs.eigs(self.EL.T, k = 1, which = 'LM') # Fixed point of E
        T = T / np.sqrt(np.abs(eta[0]))
        VL = np.reshape(VL, (self.Ds, self.Ds))
        w, vl = LA.eig(VL)
        L = np.dot(vl, np.diag(np.sqrt(w))).T # VL = L * L^dagger
        _, L = linalg.qr(L)
        L = L / LA.norm(L)
        TL = Sub.NCon([L, T, LA.inv(L)], [[-1, 1], [1, -2, 2], [2, -3]]) # Obtain the initial guess for TL

        # Solve the equation R^-1 * T * R = TR
        eta, VR = LAs.eigs(self.EL, k = 1, which = 'LM')
        VR = np.reshape(VR, (self.Ds, self.Ds))
        w, vr = LA.eig(VR)
        R = np.dot(vr, np.diag(np.sqrt(w)))
        R, _ = linalg.rq(R)
        R = R / LA.norm(R)
        TR = Sub.NCon([LA.inv(R), T, R], [[-1, 1], [1, -2, 2], [2, -3]])

        # Check whether TL/TR are left/right orthonormal
        err = LA.norm(Sub.NCon([TL, np.conj(TL)], [[1, 2, -1], [1, 2, -2]]) - np.eye(self.Ds)) + LA.norm(Sub.NCon([TR, np.conj(TR)], [[-1, 2, 1], [-2, 2, 1]]) - np.eye(self.Ds))
        if err > prec: # If the above method fails, we iteratively solve the equation L * T * L-1 = TL and R-1 * T * R = TR
            err_old = 0
            for i in range(step):
                TL, L = Sub.Mps_QRP(L, T) # -> Eq. (28)
                R, TR = Sub.Mps_LQP(T, R) # -> Eq. (29)
                L = L / LA.norm(L)
                R = R / LA.norm(R)
                err = LA.norm(np.dot(TL, L) - np.tensordot(L, T, (1, 0))) + LA.norm(np.dot(T, R) - np.tensordot(R, TR, (1, 0)))
                if err < prec or abs(err_old - err) < prec / 100:
                    break
                err_old = err


        # Canonical now, but S = L * R remains to be diagonalize

        U, S, V, Dc = Sub.SplitSvd_Lapack(np.dot(L, R), self.Ds, 0)
        self.S = S / np.sqrt(np.sum(S ** 2))
        self.C = np.diag(self.S)
        self.TL = Sub.NCon([np.conj(U).T, TL, U], [[-1, 1], [1, -2, 2], [2, -3]])
        self.TR = Sub.NCon([V, TR, np.conj(V).T], [[-1, 1], [1, -2, 2], [2, -3]])
        self.TC = np.dot(self.TL, self.C)
        self.Ds = Dc

    def MinAcC(self): # Algorithm 5
        UL, _ = linalg.polar(Sub.Group(self.TC, [[0, 1], [2]]))
        VL, _ = linalg.polar(self.C) # -> Eq. (143)
        self.TL = np.reshape(np.dot(UL, np.conj(VL.T)), (self.Ds, self.Dp, self.Ds)) # -> Eq. (145)
        
        UR, _ = linalg.polar(Sub.Group(self.TC, [[0], [1, 2]]), side='left')
        VR, _ = linalg.polar(self.C, side='left') # -> Eq. (144)
        self.TR = np.reshape(np.dot(np.conj(VR.T), UR), (self.Ds, self.Dp, self.Ds))

    def NullSpace(self): # Null space of T -> Eq. (85)
        V = linalg.null_space(Sub.Group(np.conj(self.TL), [[0, 1], [2]]).T)
        return np.reshape(V, (self.Ds, self.Dp, -1))

    def Truncation(self, Ds, iter=True, prec=1e-12, step=1000): # Algorithm 3
        if iter == True:
            T_trun = iMPS(Ds, self.Dp)
            err_old = 0
            FL = np.random.rand(self.Ds, T_trun.Ds) + 1j * np.random.rand(self.Ds, T_trun.Ds)
            FR = np.random.rand(self.Ds, T_trun.Ds) + 1j * np.random.rand(self.Ds, T_trun.Ds)
            for i in range(step):
                # Fixed point -> Eq. (99,100)
                EL, ER = self.Transfer_two(T_trun, direction = 2)
                eta, FL = LAs.eigs(EL.T, k = 1, which = 'LM', v0 = Sub.Group(FL, [[0, 1]]), tol = err_old / 100)
                FL = np.reshape(FL, (self.Ds, T_trun.Ds))
                eta, FR = LAs.eigs(ER, k = 1, which = 'LM', v0 = Sub.Group(FR, [[0, 1]]), tol = err_old / 100)
                FR = np.reshape(FR, (self.Ds, T_trun.Ds))

                # Normalize FL -> Eq. (251)
                Overlap = Sub.NCon([FL, FR], [[1, 2], [1, 2]])
                FL = FL / Overlap

                # Map O_AC/O_C -> Eq. (101, 102)
                T_trun.TC = Sub.NCon([FL, FR, self.TC], [[1, -1], [2, -3], [1, -2, 2]])
                T_trun.C = Sub.NCon([FL, FR, self.C], [[1, -1], [2, -2], [1, 2]])
                T_trun.MinAcC()
                err = (LA.norm(np.dot(T_trun.TL, T_trun.C) - T_trun.TC) + LA.norm(np.tensordot(T_trun.C, T_trun.TR, (1, 0)) - T_trun.TC))/ LA.norm(T_trun.TC)
                
                if err < prec or abs(err_old - err) < prec:
                    break
                err_old = err

            if err > 1e-9:
                logging.info('Truncation fail!')
            T_trun.Canonical()

        else:
            T_trun = copy.deepcopy(self)
            U, S, V, Dc = Sub.SplitSvd_Lapack(T_trun.C, Ds, 0) # Diagonalize C in Eq. (15)
            T_trun.S = S / np.sqrt(np.sum(S ** 2))
            T_trun.C = np.diag(T_trun.S)
            T_trun.TL = Sub.NCon([np.conj(U).T, T_trun.TL, U], [[-1, 1], [1, -2, 2], [2, -3]])
            T_trun.TR = Sub.NCon([V, T_trun.TR, np.conj(V).T], [[-1, 1], [1, -2, 2], [2, -3]])
            T_trun.TC = np.dot(T_trun.TL, T_trun.C)
            T_trun.Ds = Dc
    
        EL, ER = self.Transfer_two(T_trun, direction = 2)
        eta, FL = LAs.eigs(EL.T, k = 1, which = 'LM')

        return T_trun, eta

    def Cal_VUMPS(self, h, k, which='SR', prec=1e-12, print_err=True, space=100, step=1000): # Algorithm 4
        '''
        h is the local Hamiltonian with interaction length k, only work for k=2 now
        '''
        err_old = 0
        for r in range(step):
            self.Canonical()
            self.Transfer(direction = 2)
            ll = np.eye(self.Ds)
            lr = self.C@np.conj(self.C.T)
            rl = self.C.T@np.conj(self.C)
            rr = np.eye(self.Ds)
        
            Lh = [None]*(k+1)
            Lh[0] = np.eye(self.Ds).reshape(self.Ds, 1, self.Ds)
            for i in range(k):
                Lh[i+1] = Sub.NCon([Lh[i], self.TL, h[i], np.conj(self.TL)], [[1, 3, 2], [1, 4, -1], [4, 3, 5, -2], [2, 5, -3]])

            Rh = [None]*(k+1)
            Rh[k] = np.eye(self.Ds).reshape(self.Ds, 1, self.Ds)
            for i in range(k):
                Rh[k-i-1] = Sub.NCon([Rh[k-i], self.TR, h[k-i-1], np.conj(self.TR)], [[1, 3, 2], [-1, 4, 1], [4, -2, 5, 3], [-3, 5, 2]])

            E = Sub.NCon([Lh[k], Rh[k], self.C, np.conj(self.C)], [[1, 5, 2], [3, 5, 4], [1, 3], [2, 4]]) / Sub.NCon([ll, lr], [[1, 2], [1, 2]]) # -> Eq. (35)


            EL_tilde = self.EL - Sub.NCon([lr, ll],[[-1, -2],[-3, -4]]).reshape(self.Ds ** 2, self.Ds ** 2) / Sub.NCon([lr, ll], [[1, 2], [1, 2]]) # -> Eq. (47)
            ER_tilde = self.ER - Sub.NCon([rr, rl],[[-1, -2],[-3, -4]]).reshape(self.Ds ** 2, self.Ds ** 2) / Sub.NCon([rr, rl], [[1, 2], [1, 2]]) # -> Eq. (47)
            Pseudo_Inverse_EL = linalg.inv(np.eye(self.Ds ** 2, dtype=complex) - EL_tilde).reshape(self.Ds, self.Ds, self.Ds, self.Ds)
            Pseudo_Inverse_ER = linalg.inv(np.eye(self.Ds ** 2, dtype=complex) - ER_tilde).reshape(self.Ds, self.Ds, self.Ds, self.Ds)
            LhE = Sub.NCon([Lh[k][:, 0, :] - E * np.eye(self.Ds), Pseudo_Inverse_EL], [[1, 2], [1, 2, -1, -2]]) # -> Eq. (133)
            ERh = Sub.NCon([Rh[0][:, 0, :] - E * np.eye(self.Ds), Pseudo_Inverse_ER], [[1, 2], [-1, -2, 1, 2]]) # -> Eq. (133)

            O_TC = Sub.NCon([Lh[k-1], h[-1][:, :, :, 0], np.eye(self.Ds)], [[-4, 1, -1], [-5, 1, -2], [-3, -6]]) \
                 + Sub.NCon([np.eye(self.Ds), h[0][:, 0, :, :], Rh[1]], [[-1, -4], [-5, -2, 1], [-6, 1, -3]]) \
                 + Sub.NCon([LhE, np.eye(self.Dp), np.eye(self.Ds)], [[-4, -1], [-2, -5], [-3, -6]]) + Sub.NCon([np.eye(self.Ds), np.eye(self.Dp), ERh], [[-1, -4], [-2, -5], [-6, -3]]) # -> Eq. (131)
            O_C = Sub.NCon([Lh[1], Rh[1]], [[-3, 1, -1], [-4, 1, -2]]) + Sub.NCon([LhE, np.eye(self.Ds)], [[-3, -1], [-2, -4]]) + Sub.NCon([np.eye(self.Ds), ERh], [[-1, -3], [-4, -2]])  # -> Eq. (132)
            e, TC = LAs.eigsh(Sub.Group(O_TC, [[0, 1, 2], [3, 4, 5]]), k = 1, which = which, v0 = Sub.Group(self.TC, [[0, 1, 2]]), tol = err_old / 100)
            e, C = LAs.eigsh(Sub.Group(O_C, [[0, 1], [2, 3]]), k = 1, which = which, v0 = Sub.Group(self.C, [[0, 1]]), tol = err_old / 100)
            self.TC = np.reshape(TC, (self.Ds, self.Dp, self.Ds))
            self.C = np.reshape(C, (self.Ds, self.Ds))

            self.MinAcC()
            err = (LA.norm(np.dot(self.TL, self.C) - self.TC) + LA.norm(np.tensordot(self.C, self.TR, (1, 0)) - self.TC)) / LA.norm(self.TC)        

            if print_err == True and r%space == (space - 1):
                logging.info(r, err, E)
            if err < prec or abs(err_old - err) < prec / 100:
                break
            err_old = err

        if err > 1e-9:
            logging.info('Optimization fail!')
        
        self.Canonical()
        return self, E
    
    def Measure(self, S):
        l = np.eye(self.Ds)
        r = np.dot(self.C, np.conj(self.C.T))
        return Sub.NCon([l, self.TL, S, np.conj(self.TL), r], [[1, 2], [1, 3, 5], [4, 3], [2, 4, 6], [5, 6]]) / Sub.NCon([r, l], [[1, 2], [1, 2]])

    def __str__(self):
        return 'infinite MPS with dp = ' + str(self.Dp)+', D = ' + str(self.Ds)

    def __mul__(self, other_MPS): # Calculate <other_MPS|self> without normalization
        EL, _ = self.Transfer_two(other_MPS)
        return LAs.eigs(EL.T, k = 1, which = 'LM')[0][0]

    def __eq__(self, other_MPS):
        return (1 - abs(self * other_MPS) / np.sqrt(abs(self * self) * abs(other_MPS * other_MPS))) < 1e-12

def get_local_hamiltonian_ising(J, g):
    h = [None] * 2
    h[0] = np.zeros([2, 1, 2, 3], dtype=complex)
    h[1] = np.zeros([2, 3, 2, 1], dtype=complex)
    h[0][:, 0, :, 0] = pauli[0]
    h[0][:, 0, :, 1] = -J * pauli[3]
    h[0][:, 0, :, 2] = -g / 2 * pauli[1]
    h[1][:, 0, :, 0] = -g / 2 * pauli[1]
    h[1][:, 1, :, 0] = pauli[3]
    h[1][:, 2, :, 0] = pauli[0]
    return h

def get_local_hamiltonian_XXZ(Jx, Jz, g):
    h = [None] * 2
    h[0] = np.zeros([2, 1, 2, 4], dtype=complex)
    h[1] = np.zeros([2, 4, 2, 1], dtype=complex)
    h[0][:, 0, :, 0] = pauli[0]
    h[0][:, 0, :, 1] = -Jx * pauli[1]
    h[0][:, 0, :, 2] = -Jz * pauli[3]
    h[0][:, 0, :, 3] = -g / 2 * pauli[3]
    h[1][:, 0, :, 0] = -g / 2 * pauli[3]
    h[1][:, 1, :, 0] = pauli[1]
    h[1][:, 2, :, 0] = pauli[3]
    h[1][:, 3, :, 0] = pauli[0]
    return h

if __name__ == "__main__":
    h = get_local_hamiltonian_ising(J = 1.0, g = 1.1)
    # h = get_local_hamiltonian_XXZ(Jx = 1.0, Jz = 0.5, g = 0.1)
    imps = iMPS(10, 2)
    imps, energy = imps.Cal_VUMPS(h, k = 2, which='SR', prec=1e-12, print_err=True, space=10, step=1000)
    sigmax = imps.Measure(pauli[1])
    sigmaz = imps.Measure(pauli[3])
    print(energy.real, sigmax.real, sigmaz.real)
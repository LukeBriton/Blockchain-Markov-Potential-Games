import hashlib
import time
import numpy as np

class SimpleLedger:
    def __init__(self):
        self.blocks = []  # list of dicts

    def append(self, record: dict):
        # very lightweight "block hash chain"
        prev = self.blocks[-1]["hash"] if self.blocks else "GENESIS"
        payload = (prev + repr(record)).encode("utf-8")
        h = hashlib.sha256(payload).hexdigest()
        record2 = dict(record)
        record2["prev_hash"] = prev
        record2["hash"] = h
        self.blocks.append(record2)
        return h

class IncentiveContract:
    """
    Minimal smart-contract-like object:
    - logs (s, a, base_r)
    - computes adjusted rewards by penalties/redistribution
    - keeps token balances (optional)
    """
    def __init__(self, N, act_dic, state_dic, penalty_coef=10.0, bonus_coef=0.0):
        self.N = N
        self.act_dic = act_dic
        self.state_dic = state_dic
        self.penalty_coef = penalty_coef
        self.bonus_coef = bonus_coef
        self.ledger = SimpleLedger()
        self.balances = np.zeros(N, dtype=float)

    def log_step(self, s, actions, base_rewards, note=""):
        rec = {
            "ts": time.time(),
            "state": int(s),
            "actions": tuple(int(x) for x in actions),
            "base_rewards": tuple(float(x) for x in base_rewards),
            "note": note,
        }
        return self.ledger.append(rec)

    def detect_anomaly(self, s, actions):
        # density computed from the CongGame object
        acts = [self.act_dic[i] for i in actions]
        density = self.state_dic[s].get_counts(acts)
        max_d = max(density)
        # example thresholds mirroring your state transition logic
        thr = (self.N / 2) if s == 0 else (self.N / 4)
        if max_d <= thr:
            return np.zeros(self.N, dtype=bool), density
        # flag agents on a max-density facility
        max_facilities = {j for j, d in enumerate(density) if d == max_d}
        flagged = np.zeros(self.N, dtype=bool)
        for i, a in enumerate(actions):
            fac = self.act_dic[a][0]  # singleton facility tuple
            if fac in max_facilities:
                flagged[i] = True
        return flagged, density

    def rewardDistribution(self, s, actions, base_rewards):
        flagged, density = self.detect_anomaly(s, actions)

        adjusted = np.array(base_rewards, dtype=float)

        # Penalty: subtract if flagged
        adjusted[flagged] -= self.penalty_coef

        # Optional bonus for load balancing (e.g., negative variance of density)
        if self.bonus_coef != 0.0:
            bonus = -np.var(density)
            adjusted += self.bonus_coef * bonus / self.N

        # Update balances (token settlement)
        self.balances += adjusted

        self.log_step(s, actions, base_rewards, note=f"flagged={flagged.tolist()}, density={density}")
        return adjusted.tolist()

    def log_policy_hash(self, agent_id, state_id, policy_vec):
        import numpy as np
        b = np.asarray(policy_vec, dtype=np.float64).tobytes()
        h = hashlib.sha256(b).hexdigest()
        self.ledger.append({
            "ts": time.time(),
            "type": "policy_hash",
            "agent": int(agent_id),
            "state": int(state_id),
            "hash": h,
        })
        return h

from enum import IntEnum
import hashlib
import time
import numpy as np

class BCMode(IntEnum):
    NONE = 0        # no logging, no mechanism
    LOG_ONLY = 1    # logging only (auditability)
    FULL = 2        # logging + rewardDistribution() penalties/redistribution

class SimpleLedger:
    """Tiny in-memory append-only hash chain (simulates immutability/audit)."""
    def __init__(self, keep_records=False):
        self.keep_records = keep_records
        self.prev_hash = "GENESIS"
        self.records = [] if keep_records else None

    def append(self, payload: dict) -> str:
        b = (self.prev_hash + repr(payload)).encode("utf-8")
        h = hashlib.sha256(b).hexdigest()
        self.prev_hash = h
        if self.keep_records:
            payload2 = dict(payload)
            payload2["hash"] = h
            self.records.append(payload2)
        return h

class IncentiveContract:
    """
    "Smart contract" façade:
      - logs transitions (s, actions, rewards)  (Data on-chain) :contentReference[oaicite:1]{index=1}
      - computes rewardDistribution() settlement and penalties :contentReference[oaicite:2]{index=2}
      - tracks metrics: social welfare + anomaly rate (for ablation plots) :contentReference[oaicite:3]{index=3}
    """
    def __init__(self, N, act_dic, state_dic, mode=BCMode.NONE,
                 penalty_coef=10.0, bonus_coef=0.0, initial_balance=0.0, keep_ledger=False):
        self.N = N
        self.act_dic = act_dic
        self.state_dic = state_dic
        self.mode = BCMode(mode)
        self.penalty_coef = float(penalty_coef)
        self.bonus_coef = bonus_coef
        self.ledger = SimpleLedger(keep_records=keep_ledger)
        # On-chain “account balances” (paper: initial token balance, then contract adjusts balances) :contentReference[oaicite:1]{index=1}
        self.balances = np.full(N, float(initial_balance), dtype=float)

        # Metrics (per run)
        self.reset_metrics()

    def log_step(self, s: int, actions, base_rewards, note=""):
        payload = {
            "ts": time.time(),
            "state": int(s),
            "actions": tuple(int(x) for x in actions),
            "base_rewards": tuple(float(x) for x in base_rewards),
            "note": note,
        }
        self.ledger.append(payload)

    def reset_metrics(self):
        self.steps = 0
        self.anomaly_steps = 0
        self.welfare_sum = 0.0

    def _detect_anomaly(self, s: int, actions):
        """Example detection rule matching your transition thresholds."""
        # density computed from the CongGame object
        acts = [self.act_dic[i] for i in actions]              # e.g. (facility,)
        density = self.state_dic[s].get_counts(acts)           # list length D
        max_d = max(density)
        # example thresholds mirroring your state transition logic
        thr = (self.N / 2.0) if s == 0 else (self.N / 4.0)
        anomalous = (max_d > thr)
        # flag agents on a max-density facility
        max_facilities = {j for j, d in enumerate(density) if d == max_d}
        flagged = np.zeros(self.N, dtype=bool)
        if anomalous:
            # penalize agents on the maximally congested facility/facilities
            for i, a_int in enumerate(actions):
                fac = self.act_dic[a_int][0]  # singleton facility tuple -> int
                if fac in max_facilities:
                    flagged[i] = True
        return anomalous, flagged.tolist(), density



    def rewardDistribution(self, s: int, actions, base_rewards,
                           record=True,            # count steps + metrics?
                           update_balances=True,   # update token balances?
                           do_log=True):           # append ledger record?
        """
        Settlement step: returns either base rewards (LOG_ONLY) or adjusted (FULL),
        and logs. This mirrors the paper's 'rewardDistribution() aggregates data
        and computes payoffs/penalties' description. :contentReference[oaicite:4]{index=4}
        """
        base = np.array(base_rewards, dtype=float)

        anomalous, flagged, density = self._detect_anomaly(s, actions)

        if self.mode == BCMode.FULL and anomalous:
            # 1) penalize flagged agents (confiscation-like)
            adjusted = base.copy()
            for i, f in enumerate(flagged):
                if f:
                    adjusted[i] -= self.penalty_coef
            # 2) optional: redistribute confiscated amount equally to non-flagged
            # (keeps total welfare similar, but shifts incentive)
            pool = np.sum(base - adjusted)  # total penalty collected (>=0)
            if pool > 0:
                winners = [i for i, f in enumerate(flagged) if not f]
                if winners:
                    adjusted[winners] += pool / len(winners)
        else:
            adjusted = base  # NONE or LOG_ONLY does not change rewards

        # 3) optional global load-balancing bonus (same for everyone)
        if self.bonus_coef != 0.0:
            bonus = - self.bonus_coef * np.var(density) # or -bonus_coef * max(density)
            adjusted += bonus / self.N                  # equal share to each agent

        # 4) update on-chain balances (accounting, token settlement)
        if update_balances:
            self.balances += adjusted

        # 5) Metrics (for ablation comparison: welfare + anomaly rate)
        if record:
            self.steps += 1
            if anomalous:
                self.anomaly_steps += 1
            self.welfare_sum += float(np.sum(adjusted))

        # 6) Logging (auditability, for LOG_ONLY / FULL)
        if do_log and self.mode in (BCMode.LOG_ONLY, BCMode.FULL):
            self.log_step(s, actions, adjusted.tolist(),
                          note=f"anomalous={anomalous}, flagged={flagged}, density={density}")

        return adjusted.tolist()

    def log_policy_hash(self, agent_id: int, state_id: int, policy_vec):
        """
        Optional: hash of policy update stored "on-chain" for verifiability. :contentReference[oaicite:6]{index=6}
        """
        b = np.asarray(policy_vec, dtype=np.float64).tobytes()
        h = hashlib.sha256(b).hexdigest()
        if self.mode in (BCMode.LOG_ONLY, BCMode.FULL):
            self.ledger.append({
                "ts": time.time(),
                "type": "policy_hash",
                "agent": int(agent_id),
                "state": int(state_id),
                "hash": h,
            })
        return h

    def metrics(self):
        anomaly_rate = (self.anomaly_steps / self.steps) if self.steps else 0.0
        welfare_per_step = (self.welfare_sum / self.steps) if self.steps else 0.0
        return {
            "steps": self.steps,
            "anomaly_rate": anomaly_rate,
            "welfare_per_step": welfare_per_step,
        }

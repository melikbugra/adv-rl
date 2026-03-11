"""Check adversarial training progress via MLflow."""

import datetime
import mlflow

TRACKING_URI = "https://mlflow.melikbugraozcelik.com/"
ADV_EXP_NAME = "FetchPegInHolePreHeldDense-v1_adv-sac_(q-ensemble_reward,_peg-in-hole)"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    exps = client.search_experiments()
    for e in exps:
        if "adv" in e.name.lower() or ("peg" in e.name.lower() and "sac" not in e.name.lower()):
            print(f"\nExperiment: {e.name}")
            runs = client.search_runs(
                e.experiment_id,
                order_by=["start_time DESC"],
                max_results=5,
            )
            for r in runs:
                t = datetime.datetime.fromtimestamp(r.info.start_time / 1000)
                print(f"\n  Run : {r.info.run_name}")
                print(f"  Status : {r.info.status}")
                print(f"  Started: {t:%Y-%m-%d %H:%M:%S}")

                m = r.data.metrics
                iteration = m.get("iteration")
                if iteration is not None:
                    print(f"  Iteration: {int(iteration)} / 1000  ({iteration/10:.1f}%)")

                for key in [
                    "prt_train_prt_score",
                    "adv_train_adv_score",
                    "prt_train_d_xy",
                    "prt_train_d_z",
                    "adv_train_d_xy",
                    "adv_train_d_z",
                    "adv_train_reward_depth",
                    "prt_train_reward_depth",
                    "lambda_v_eff",
                    "lambda_sigma_eff",
                ]:
                    val = m.get(key)
                    if val is not None:
                        print(f"  {key:<35s}: {val:.4f}")


if __name__ == "__main__":
    main()

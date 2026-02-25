"""Check SAC training progress for FetchStirDense-v1 via MLflow."""

import datetime
import mlflow

TRACKING_URI = "https://mlflow.melikbugraozcelik.com/"
EXP_NAME = "FetchStirDense-v1_sac"
TOTAL_TIMESTEPS = 5_000_000


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()

    exp = client.get_experiment_by_name(EXP_NAME)
    if exp is None:
        print(f"Experiment '{EXP_NAME}' not found.")
        return

    runs = client.search_runs(
        exp.experiment_id,
        order_by=["start_time DESC"],
        max_results=5,
    )

    print(f"\nExperiment: {exp.name}")
    print("=" * 60)

    for r in runs:
        t = datetime.datetime.fromtimestamp(r.info.start_time / 1000)
        print(f"\n  Run    : {r.info.run_name}")
        print(f"  Status : {r.info.status}")
        print(f"  Started: {t:%Y-%m-%d %H:%M:%S}")

        m = r.data.metrics

        train_score = m.get("Train Score")
        eval_score  = m.get("Average Evaluation Score")
        step        = m.get("step") or m.get("timestep")

        if train_score is not None:
            print(f"  Train Score      : {train_score:.2f}  (max scale ~880)")
        if eval_score is not None:
            print(f"  Eval Score       : {eval_score:.2f}")

        if step is not None:
            pct = step / TOTAL_TIMESTEPS * 100
            print(f"  Timestep         : {int(step):,} / {TOTAL_TIMESTEPS:,}  ({pct:.1f}%)")

        for key in [
            "Average Actor Loss",
            "Average Critic Loss",
            "Average Alpha Loss",
        ]:
            val = m.get(key)
            if val is not None:
                print(f"  {key:<25s}: {val:.4f}")


if __name__ == "__main__":
    main()

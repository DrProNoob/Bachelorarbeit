import logging
import os
import time
from pathlib import Path

import pandas as pd
from openai import RateLimitError
from kag.solver.logic.solver_pipeline import SolverPipeline
from kag.common.conf import KAG_CONFIG


logger = logging.getLogger(__name__)

QUESTION_TEMPLATES = [
    "Does the {contract} contract contain an exception or carve‑out to any competitive restriction, allowing certain competitive activities or partnerships that would otherwise be prohibited, and if so, does it specify the activities or partnerships allowed?",
    "Does the {contract} contract include a clause that restricts a party from competing or engaging in a business activity that competes with the other party, and if so, does it specify a time frame or geographic area for this restriction?",
    "Does the {contract} contract grant one party exclusive rights to provide certain goods or services, and if so, does it also prevent that party from entering into similar agreements with third parties?",
    "Is there a provision in the {contract} contract that prohibits a party from soliciting or doing business with the customers or clients of the other party, and if so, does it specify a duration for this prohibition?"
]


class EvaForBa:
    #pfad ändern
    def __init__(self, data_dir: str = "/Users/danielmentjukov/Downloads/kag/KAG/kag/examples/KagBATest/builder/data"):
        self.data_dir = data_dir
        self.solver_pipeline = SolverPipeline.from_config(
            KAG_CONFIG.all_config["kag_solver_pipeline"]
        )
        pass

    def _prepare_questions(self):
        question_templates = QUESTION_TEMPLATES
        data_dir = self.data_dir

        contracts = [
            f.replace(".txt", "")
            for f in os.listdir(data_dir)
            if f.endswith(".txt")
        ]
        logger.info(f"Anzahl der Verträge gefunden: {len(contracts)}")
        multi_hop_q = []
        for contract in contracts:
            for template in question_templates:
                question = template.format(contract=contract)
                multi_hop_q.append({"question": question, "contract": contract})
        logger.info(
            f"Gesamtzahl der initial generierten Fragen: {len(multi_hop_q)}"
        )
        return multi_hop_q

    def _ask_with_retry(self, prompt: str) -> str:
        max_retries = 3
        wait_seconds = 10
        for attempt in range(max_retries + 1):
            try:
                # solver_pipeline.run() liefert (answer, tracelog)
                result = self.solver_pipeline.run(prompt)

                # Nur die eigentliche Antwort zurückgeben, den Trace-Log verwerfen
                if isinstance(result, tuple) and len(result) >= 1:
                    answer, *_ = result
                else:
                    answer = result

                return answer

            except RateLimitError as e:
                if attempt == max_retries:
                    return f"RateLimitError: {e}"
                logger.warning(
                    f"Rate‑Limit, versuche erneut in {wait_seconds}s …"
                )
                time.sleep(wait_seconds)
            except Exception as e:
                return f"ERROR: {e}"

    def _process_item(self, item: dict) -> dict:
        prompt = (
            "Answer in two parts: (1) Yes/No for the first condition, "
            "(2) details for the second condition. "
            + item["question"]
        )
        answer = self._ask_with_retry(prompt)
        status = (
            "success"
            if not answer.startswith(("RateLimitError", "ERROR"))
            else "failed"
        )
        return {
            "contract": item["contract"],
            "question": item["question"],
            "kag_answer": answer,
            "status": status,
        }

    @staticmethod
    def _save_rows(rows: list[dict], path: str):
        df = pd.DataFrame(
            rows, columns=["contract", "question", "kag_answer", "status"]
        )
        first_write = not Path(path).exists()
        df.to_csv(
            path,
            mode="a" if not first_write else "w",
            header=first_write,
            index=False,
        )

    @staticmethod
    def _load_failed(path: str) -> list[dict]:
        if not Path(path).exists():
            logger.warning(f"Datei {path} nicht gefunden – kein Retry möglich.")
            return []
        df = pd.read_csv(path)
        return df[df.get("status") != "success"].to_dict("records")


    def initial_run(self, output_file: str = "./final_result.csv"):
        results = [self._process_item(q) for q in self._prepare_questions()]
        self._save_rows(results, output_file)
        logger.info(
            f"Initialer Lauf fertig – {len(results)} Zeilen gespeichert."
        )

    def retry_run(
        self,
        previous_file: str = "./final_result.csv",
        output_file: str = "./final_combined_resultT.csv",
    ):
        retry_items = self._load_failed(previous_file)
        if not retry_items:
            logger.info("Keine fehlgeschlagenen Zeilen – Retry übersprungen.")
            return
        retry_results = [self._process_item(it) for it in retry_items]

        # alte neue ergebnisse zusammenführen
        old_df = pd.read_csv(previous_file) if Path(previous_file).exists() else pd.DataFrame()
        combined = pd.concat(
            [old_df[old_df.get("status") == "success"], pd.DataFrame(retry_results)]
        )
        combined = combined.drop_duplicates(subset=["contract", "question"])
        combined.to_csv(output_file, index=False)
        logger.info(
            f"Retry fertig – Gesamt: {len(combined)} Zeilen in {output_file}."
        )
    


if __name__ == "__main__":
    eva = EvaForBa()

    INITIAL_RUN = True  

    if INITIAL_RUN:
        eva.initial_run()
    else:
        eva.retry_run()





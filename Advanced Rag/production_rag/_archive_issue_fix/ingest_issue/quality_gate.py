"""[ISSUE version] quality_gate.py - STAGE 4 of ingestion: check the chunks before anything is embedded or stored.

For a PDF the rule is simple: every number in the original PDF must appear in the final chunks.
If one is missing, the file is rejected (not indexed) and we say which numbers are missing.
Rates, fees and limits are numbers. A missing number is a wrong answer waiting to happen.
"""


def check_chunks(chunks):

    # ===== ISSUE T2: no check. Whatever the chunks contain goes on to the database, even if numbers were lost =====
    accepted = chunks

    print(f"  {len(accepted)} of {len(chunks)} chunks go on to the next stage")
    return accepted

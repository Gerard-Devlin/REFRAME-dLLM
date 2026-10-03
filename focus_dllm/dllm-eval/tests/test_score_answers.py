import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from dllm_eval.score_answers import (assess, audit_sample, boxes, clean_expression,
                                    extract_final, numeric_value, policy_hash, load_jobs)
from dllm_eval.score_math import sha256


class ExtractionTests(unittest.TestCase):
    def test_nested_latex_and_escaped_braces(self):
        result = extract_final(r"We conclude \[\boxed{\frac{1}{2}}\]")
        self.assertEqual(result["candidates"][0]["answer"], r"\frac{1}{2}")
        self.assertEqual(boxes(r"\boxed{\{1,2\}}")[-1]["answer"], r"\{1,2\}")

    def test_wrong_final_with_correct_intermediate(self):
        for text in ("Calculation: 42. Final answer: 43", r"Earlier \boxed{42}. The answer is 43.",
                     "#### 42\nCorrection. Final answer: 43"):
            self.assertFalse(assess(text,"42","gsm8k")["correct"])
            self.assertEqual(assess(text,"42","gsm8k")["prediction"],"43")

    def test_supported_formats(self):
        for text in (r"\boxed{42}", "#### 42", "The final answer is 42.",
                     "Final Answer:\n\n42", r"Answer: \(42\)", "42"):
            self.assertTrue(assess(text,"42","gsm8k")["correct"],text)

    def test_malformed_final_never_reuses_earlier_correct(self):
        text = r"Earlier \boxed{42}. Final answer: \boxed{43"
        result = assess(text,"42","gsm8k")
        self.assertEqual(result["status"],"malformed")
        self.assertFalse(result["correct"])

    def test_conflicting_final_boxes(self):
        result = assess(r"Final answer: \boxed{42} or \boxed{43}","42","gsm8k")
        self.assertEqual(result["status"],"ambiguous")
        self.assertFalse(result["correct"])

    def test_duplicate_equivalent_answers(self):
        self.assertTrue(assess(r"Final answer: \boxed{1/2}, \boxed{0.5}","0.5","gsm8k")["correct"])

    def test_multiple_final_math_environments(self):
        result = assess(r"Final answer: $42$ or $43$", "42", "gsm8k")
        self.assertEqual(result["status"], "ambiguous")

    def test_does_not_search_prose_for_last_number(self):
        self.assertEqual(extract_final("We tried 42 and then 43.")["status"],"missing")
        self.assertFalse(assess("We tried 42 and then 43.","43","gsm8k")["correct"])

    def test_fraction_decimal_comma_currency(self):
        for text, gold in (("Answer: 1/2","0.5"),("Answer: $1,200 dollars.","1200"),
                           ("Answer: 12.5%", "12.5"),("Answer: -0.25", "-1/4")):
            self.assertTrue(assess(text,gold,"gsm8k")["correct"],text)
        self.assertFalse(assess("Answer: 0.333333","1/3","gsm8k")["correct"])

    def test_no_eval_or_substring_parsing(self):
        for answer in ("1+", "0/0", "__import__('os').system('touch foo')", "42 or 43"):
            self.assertIsNone(numeric_value(answer,gsm8k=True))

    def test_empty_unknown_and_weak_tail(self):
        for text in (None,"","Answer:"):
            self.assertFalse(assess(text,"42","gsm8k")["correct"])
        result = assess("Working...\n42","42","gsm8k")
        self.assertTrue(result["correct"])
        self.assertFalse(result["explicit_correct"])

    def test_extraction_does_not_depend_on_gold(self):
        text = "Answer: 43"
        self.assertEqual(assess(text,"42","gsm8k")["extraction"],
                         assess(text,"43","gsm8k")["extraction"])
        self.assertEqual(policy_hash(),policy_hash())

    def test_outer_delimiter_repair_preserves_payload(self):
        for value in (r"\(-4, 4$",r"$-4, 4\)"):
            self.assertEqual(clean_expression(value),"-4, 4")
        self.assertFalse(assess(r"Earlier \boxed{42}. Final answer: \(43$","42","gsm8k")["correct"])

    def test_gsm_natural_final_conclusions(self):
        for text,gold in (("Work: 24*10=240.\n\nSo, Stetson gave up $240.","240"),
                          ("Therefore, ten sharks have a total of 1200 gallons.","1200"),
                          ("#### 50 people","50"),
                          ("Answer: 1/2 cups","0.5"),
                          (r"Answer: \frac{1}{2}","0.5"),
                          ("Therefore, the total is 16-3-4=9 people.","9")):
            self.assertTrue(assess(text,gold,"gsm8k")["correct"],text)

    def test_gsm_conclusions_do_not_search_intermediate_or_choose_last_quantity(self):
        self.assertFalse(assess("Earlier: 42.\n\nTherefore, the answer is 43 apples.","42","gsm8k")["correct"])
        for text in ("Therefore, there are 42 or 43 apples.",
                     "Therefore, after 3 days there are 3430 people.",
                     "We tried 42 and then 43.", "Answer: not 42."):
            self.assertFalse(assess(text,"42","gsm8k")["correct"],text)

    def test_audit_is_blinded_and_deterministic(self):
        d=[dict(id="x",gold="1",methods={"ours":assess("Answer: 2","1","gsm8k"),
                                       "baseline":assess("Answer: 1","1","gsm8k")})]
        text={"ours":{"x":"Answer: 2"},"baseline":{"x":"Answer: 1"}}
        prior={"ours":{"x":{"math_verify":True}},"baseline":{"x":{"math_verify":True}}}
        audit,key=audit_sample(d,text,prior)
        self.assertEqual((audit,key),audit_sample(d,text,prior))
        self.assertNotIn("ours",json.dumps(audit))
        self.assertNotIn("baseline",json.dumps(audit))
        self.assertEqual(audit["eligible_prompt_count"],1)


class PairedInputTests(unittest.TestCase):
    def fixture(self,root):
        dataset=root/"dataset.json"
        dataset.write_text(json.dumps([dict(id="q",answer="work\n#### 42")]))
        output=root/"output";output.mkdir()
        summary=dict(dataset_sha256=sha256(dataset),model="fixed",revision="pinned",ids=["q"],
            configuration=dict(task="gsm8k",gen_length=256,block_length=32,methods=["native"]),
            results=dict(native=dict(mean_seconds=1.0)))
        (output/"summary.json").write_text(json.dumps(summary))
        (output/"rank_0.jsonl").write_text(json.dumps(dict(index=0,id="q",native=dict(text="Answer: 42")))+"\n")
        return dataset,output,summary

    def test_inputs_untouched_and_original_reference(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset,output,_=self.fixture(Path(temporary))
            rows,texts,old,provenance,hashes,samples=load_jobs(["a="+str(output)],dataset,"gsm8k")
            self.assertEqual(rows[0][1],"42")
            self.assertEqual(rows[0][3],{"a":"Answer: 42"})
            self.assertTrue(all(sha256(p)==h for p,h in hashes.items()))

    def test_duplicate_jobs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset,output,_=self.fixture(Path(temporary))
            with self.assertRaises(ValueError):
                load_jobs(["a="+str(output),"a="+str(output)],dataset,"gsm8k")

    def test_dataset_and_sample_set_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset,output,summary=self.fixture(Path(temporary))
            summary["ids"]=["different"]
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_jobs(["a="+str(output)],dataset,"gsm8k")
            summary["dataset_sha256"]="wrong"
            (output/"summary.json").write_text(json.dumps(summary))
            with self.assertRaises(ValueError):
                load_jobs(["a="+str(output)],dataset,"gsm8k")


@unittest.skipUnless(importlib.util.find_spec("math_verify"), "uses existing server Math-Verify")
class MathEquivalenceTests(unittest.TestCase):
    def test_equivalent_forms_and_wrong_final(self):
        for text in (r"\boxed{\frac{1}{2}}", r"Final answer: $0.5$", "Answer: 1/2"):
            self.assertTrue(assess(text,r"\frac{1}{2}","math")["correct"],text)
        self.assertFalse(assess(r"Working: $\frac{1}{2}$. Final answer: $\frac{1}{3}$",
                               r"\frac{1}{2}","math")["correct"])

    def test_symbolic_equivalence(self):
        self.assertTrue(assess(r"\boxed{(x+1)^2}",r"x^2+2x+1","math")["correct"])
        self.assertFalse(assess(r"\boxed{(x+1)^2}",r"x^2+1","math")["correct"])

    def test_partial_parse_and_conflict(self):
        self.assertEqual(assess("Final answer: 1+", "1", "math")["status"], "unparseable")
        self.assertEqual(assess(r"Final answer: \boxed{1/2} or \boxed{1/3}","1/2","math")["status"],"ambiguous")

    def test_numeric_units_and_percent(self):
        self.assertTrue(assess("Answer: 2 cm", "2", "math")["correct"])
        self.assertTrue(assess("Answer: 25%", r"\frac{1}{4}","math")["correct"])

    def test_complete_set_required(self):
        for text in ("Final answer: $-4$ and $4$",r"Final answer: \boxed{-4} or \boxed{4}",
                     r"Final answer: \(-4, 4$",r"Final answer: -4\text{ and }4"):
            self.assertTrue(assess(text,"-4,4","math")["correct"],text)
        self.assertFalse(assess("Final answer: $-4$ and $4$","-4","math")["correct"])
        self.assertFalse(assess("Final answer: $-4$ and $4$","-4,5","math")["correct"])
        self.assertFalse(assess("Final answer: $-4$","-4,4","math")["correct"])
        self.assertFalse(assess(r"\boxed{-4,4}","-4","math")["correct"])


if __name__ == "__main__":
    unittest.main()

import json,csv,os
root='artifacts/bloom_dataset_audit'
phase=root+'/phase3_semantic_validation'
src=root+'/extracted_dataset/source_extraction.jsonl'
rows=[json.loads(x) for x in open(src,encoding='utf-8')]
by={r['example_id']:r for r in rows}
# (level, cognitive reason, preservation, self-contained, technical note, status, reason, correction status, eligibility)
reviews={
516:[
('Remember','Retrieves the definition and named elements of misrepresentation.','Keeps the legal concept and its elements; asks general doctrine beyond the scenario.','Yes','General legal terminology; jurisdiction-specific doctrine may vary.','REVIEW','Legal elements depend on jurisdiction; no legal framework is named.','UNCHANGED','PENDING'),
('Understand','Explains the causal relevance of the seller’s knowingly false statement to the buyer’s decision.','Retains vehicle history, seller knowledge, and buyer decision context.','Yes','No specific technical error; legal implications remain jurisdiction-dependent.','PASS','','UNCHANGED','APPROVED'),
('Apply','Maps the supplied facts to stated general elements.','Retains all scenario facts and the misrepresentation concept.','Partial','Application is feasible only under an assumed general framework; exact legal elements vary.','REVIEW','No jurisdiction or governing rule is supplied.','UNCHANGED','PENDING'),
('Evaluate','Requires a judgment under an unspecified “relevant legal framework.”','Scenario retained, but the decisive standard is external and unspecified.','No','Cannot determine actionability without jurisdiction-specific criteria.','REVIEW','The question invokes a framework not provided.','UNCHANGED','PENDING'),
('Create','Constructs a structured argument from the stated transaction facts.','Preserves the scenario and central legal issue.','Partial','An argument can be constructed, but legal conclusions depend on jurisdiction.','REVIEW','Subject-matter review needed to establish governing legal elements.','UNCHANGED','PENDING')],
517:[
('Remember','Recalls the stated equation and interprets its symbols.','Retains the cooling model and parameter meanings.','Yes','Equation is stated; k has units min⁻¹.','PASS','','UNCHANGED','APPROVED'),
('Understand','Interprets how initial and ambient temperatures and k shape change over time.','Retains the exact parameters and cooling model.','Yes','Conceptually sound.','PASS','','UNCHANGED','APPROVED'),
('Analyze','Examines the exponential decay term and its relation to the object–ambient temperature difference.','Retains the specified 10-minute interval and model.','Yes','Sound for Newton cooling with constant k and ambient temperature.','PASS','','UNCHANGED','APPROVED'),
('Apply','Substitutes supplied values and computes T(10)=30+60e⁻¹≈52.07°C; “verify” adds only numerical checking.','All parameters and units retained.','Yes','Reported value 52.07°C is correct to two decimals.','MAJOR','A numerical verification is procedural application, not a criteria-based judgment.','REVIEW_REQUIRED','NOT_APPROVED'),
('Apply','Substitutes values and carries out the stated calculation; this is execution of a formula.','Preserves the exact calculation and conditions.','Yes','No issue; expected result is approximately 52.07°C.','MAJOR','A worked calculation is Apply rather than meaningful construction/Create.','REVIEW_REQUIRED','NOT_APPROVED')],
518:[
('Remember','Recalls the three cash-flow categories and their general meanings.','Retains the statement-of-cash-flows topic but drops the five transaction details; suitable as a narrower recall prompt.','Yes','Category terminology is broadly sound; indirect method affects operating section presentation, not basic activity categories.','PASS','','UNCHANGED','APPROVED'),
('Understand','Explains why the five transactions map to different cash-flow activities.','All five transactions retained.','Yes','Classification concepts are standard.','PASS','','UNCHANGED','APPROVED'),
('Analyze','Examines transaction purposes to distinguish the three activity groups.','All five transactions retained; same objective.','Yes','No specific error identified.','PASS','','UNCHANGED','APPROVED'),
('Apply','Checks a proposed mapping against the supplied transaction list; determining classification is procedural application, not evaluation.','Includes all transactions and proposed categories.','Yes','Proposed mappings are correct under conventional cash flow classification.','MAJOR','Checking the classification executes the classification rule; no evaluative criteria are used.','REVIEW_REQUIRED','NOT_APPROVED'),
('Apply','Categorizes listed amounts into statement headings; this is routine classification/presentation.','Preserves all amounts and activity categories.','Yes','Amounts can be grouped; format alone does not create a new artifact.','MAJOR','Constructing a categorized extract is routine application, not meaningful Create.','REVIEW_REQUIRED','NOT_APPROVED')],
519:[
('Remember','Recalls definitions of cross-ventilation and named design factors.','Retains topic and listed variables, but is a broad factual recall prompt.','Yes','No specific technical error identified.','PASS','','UNCHANGED','APPROVED'),
('Understand','Explains how opening positions establish a flow path through the classroom.','Retains opening relationship and classroom context.','Yes','General conceptual question.','PASS','','UNCHANGED','APPROVED'),
('Apply','Selects an arrangement using an assumed prevailing-wind direction and opposite available wall.','Retains classroom ventilation topic, but adds boundary conditions not present in source.','Partial','Could answer with common design principles, though “suitable” has no criteria and actual wind/building constraints are absent.','MINOR','Adds an opposite-wall condition; selection criterion is not explicit.','REVIEW_REQUIRED','PENDING'),
('Evaluate','Judges an opposite-wall arrangement using wind direction and airflow through occupied space.','Same topic and factors; no measured performance evidence or criteria supplied.','Partial','Effectiveness depends on wind pressure, openings, obstructions, and comfort criteria.','REVIEW','Insufficient design data and no explicit evaluation standard.','UNCHANGED','PENDING'),
('Create','Designs a conceptual layout with wind, openings, and occupied-zone flow path.','Preserves same classroom and design variables; actual room geometry is not required for conceptual output.','Yes','No numerical or code claim.','PASS','','UNCHANGED','APPROVED')],
520:[
('Remember','Recalls the definitions of statement and branch coverage.','Retains the two central coverage measures, but omits code/input specifics.','Yes','Definitions are standard.','PASS','','UNCHANGED','APPROVED'),
('Understand','Interprets how positive and negative test inputs traverse the if and else paths.','Retains code and both supplied test values.','Yes','Both outcomes are exercised.','PASS','','UNCHANGED','APPROVED'),
('Apply','Executes coverage definitions: both assignments and both branches run, yielding 100% statement and branch coverage.','Exact code and tests retained.','Yes','Correct for this two-branch snippet under conventional statement/branch coverage.','PASS','','UNCHANGED','APPROVED'),
('Evaluate','Judges completeness of the test set against the explicit criterion of complete statement and branch coverage, with path evidence.','Exact program and inputs retained.','Yes','Both branches and assignments are exercised; the conclusion is supported by coverage criteria.','PASS','','UNCHANGED','APPROVED'),
('Apply','Maps each input to its executed branch and assignment; this is execution tracing.','Exact code and values retained.','Yes','x=5 takes then; x=-3 takes else.','MAJOR','A coverage trace is routine application, not a meaningfully new artifact/Create.','REVIEW_REQUIRED','NOT_APPROVED')]
}
fields=['example_id','discipline_course_topic','source_question','source_bloom_declared','target_bloom_declared','transformed_question','target_bloom_assessed','actual_cognitive_demand_reasoning','content_preservation','self_containment','technical_correctness','status','status_reason','audit_state','correction_status','training_eligibility']
out=[]
for eid, assessments in reviews.items():
    r=by[eid]
    for t,a in zip(r['transformations'],assessments):
        level,reason,pres,sc,tech,status,why,corr,elig=a
        out.append(dict(example_id=eid,discipline_course_topic=r['discipline_course_topic'],source_question=r['original_question'],source_bloom_declared=r['original_bloom'],target_bloom_declared=t['target_bloom'],transformed_question=t['transformed_question'],target_bloom_assessed=level,actual_cognitive_demand_reasoning=reason,content_preservation=pres,self_containment=sc,technical_correctness=tech,status=status,status_reason=why,audit_state='Independent semantic review batch 516-520',correction_status=corr,training_eligibility=elig))
file=phase+'/transformation_batch_examples_516_520.csv'
with open(file,'w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
print(file, len(out), 'rows; statuses', {s:sum(x['status']==s for x in out) for s in ['PASS','MINOR','MAJOR','REJECT','REVIEW']})

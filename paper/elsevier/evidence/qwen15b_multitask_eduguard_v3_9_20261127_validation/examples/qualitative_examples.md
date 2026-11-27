# Qualitative examples (validation)

Selected deterministically from `lora` predictions: for QA and summarization the highest, median and lowest per-example ROUGE-L; for Bloom rewriting a seeded random sample of passing and failing rewrites. Sources are clipped for length.

## Closed-book QA: highest ROUGE-L

- ID: `QA-0827`
- ROUGE-L: 0.741

**Question.** What is the gauge pressure at a depth of 3 m in water, using density 1000 kg/m³ and g=9.8 m/s²?

**Reference.** Gauge pressure is p=ρgh=1000×9.8×3=29,400 Pa, or 29.4 kPa.

**EduGuard output.** Gauge pressure is p = ρgh = 1000 × 9.8 × 3 = 29400 Pa.

## Closed-book QA: median ROUGE-L

- ID: `QA-1363`
- ROUGE-L: 0.242

**Question.** Why are shear-force and bending-moment diagrams prepared when analyzing a simply supported beam?

**Reference.** The diagrams show how internal shear force and bending moment vary along the beam. They help identify critical sections and maximum values for structural design.

**EduGuard output.** Shear force indicates where internal forces change direction along the span, while bending moment shows how external loads cause curvature in the beam's cross-section. These diagrams support locating critical points such as supports, maximum moments, and neutral-axis positions.

## Closed-book QA: lowest ROUGE-L

- ID: `QA-1297`
- ROUGE-L: 0.111

**Question.** Why can wheel odometry become inaccurate when a mobile robot moves over a slippery floor?

**Reference.** Wheel odometry estimates motion from wheel rotation, assuming a predictable relationship between wheel movement and ground displacement. Slippage violates this assumption and causes accumulated position error.

**EduGuard output.** Slippery surfaces may cause wheels to slip, reduce friction, and produce less accurate velocity measurements. The resulting errors propagate through kinematic equations of motion and affect pose estimation.

## Summarization: highest ROUGE-L

- ID: `OSX-SUM-0792`
- ROUGE-L: 0.570

**Source.** Show Me the Money

What is money, what are its characteristics and functions, and what are the three parts of the U.S. money supply?

Money is anything that is acceptable as payment for goods and services. It affects our lives in many ways. We earn it, spend it, save it, invest it—making decisions on how to best plan for the future. Businesses and government use money in similar ways. Both require money to finance their operations. By controlling the amount of money in circulation, the federal government can promote economic growth and stability. For this reason, money has been called the oil that keeps our economic system running smoothly. Our banking system was developed to ease the handling of money.

Characteristics of Money

For money to be a suitable means of exchange, it should have these key characteristics:

Scarcity: Money should be scarce enough to have some value but not so scarce as to be unavailable. Small rocks as currency, for example, which meet some of the other criteria, would not work well as money because they are widely available. Too much money in circulation increases prices and inflation. Governments control the scarcity of money by limiting the quantity of [... 4797 more characters]

**Reference.** What is money, what are its characteristics and functions, and what are the three parts of the U.S. money supply? Money is anything accepted as payment for goods and services. For money to be a suitable means of exchange, it should be scarce, durable, portable, and divisible. Money functions as a medium of exchange, a standard of value, and a store of value. The U.S. money supply consists of currency (coins and paper money), demand deposits (checking accounts), and time deposits (interest-bearing deposits that cannot be withdrawn on demand).

**EduGuard output.** Money is anything that is accepted as payment for goods and services. Its characteristics include scarcity, durability, portability, and divisibility. Money serves as a medium of exchange, a standard of value, and a store of value. The U.S. money supply consists of currency, demand deposits, and time deposits. Currency includes coins and paper money. Demand deposits include checking accounts and other types of accounts that allow withdrawals on demand. Time deposits include savings accounts and money market deposit accounts.

## Summarization: median ROUGE-L

- ID: `OSX-SUM-0811`
- ROUGE-L: 0.224

**Source.** Pathogen Recognition and Phagocytosis

Several of the cell types discussed in the previous section can be described as phagocytes—cells whose main function is to seek, ingest, and kill pathogens. This process, called phagocytosis, was first observed in starfish in the 1880s by Nobel Prize-winning zoologist Ilya Metchnikoff (1845–1916), who made the connection to white blood cells (WBCs) in humans and other animals. At the time, Pasteur and other scientists believed that WBCs were spreading pathogens rather than killing them (which is true for some diseases, such as tuberculosis). But in most cases, phagocytes provide a strong, swift, and effective defense against a broad range of microbes, making them a critical component of innate nonspecific immunity. This section will focus on the mechanisms by which phagocytes are able to seek, recognize, and destroy pathogens.

Extravasation (Diapedesis) of Leukocytes

Some phagocytes are leukocytes (WBCs) that normally circulate in the bloodstream. To reach pathogens located in infected tissue, leukocytes must pass through the walls of small capillary blood vessels within tissues. This process, called extravasation, or diapedesis, is initiate [... 6603 more characters]

**Reference.** Phagocytes are cells that recognize pathogens and destroy them through phagocytosis. Recognition often takes place by the use of phagocyte receptors that bind molecules commonly found on pathogens, known as pathogen-associated molecular patterns (PAMPs). The receptors that bind PAMPs are called pattern recognition receptors, or PRRs. Toll-like receptors (TLRs) are one type of PRR found on phagocytes. Extravasation of white blood cells from the bloodstream into infected tissue occurs through the process of transendothelial migration. Phagocytes degrade pathogens through phagocytosis, which involves engulfing the pathogen, killing and digesting it within a phagolysosome, and then excreting undigested matter.

**EduGuard output.** Phagocytes are specialized cells that can be recruited to sites of infection via extracellular signaling pathways involving complement components, cytokines, and chemokines. Once recruited, phagocytes migrate across the endothelium of blood vessels using rolling adhesion followed by transmigration. They then engulf pathogens in a process called phagocytosis. Pathogens are recognized by pattern recognition receptors (PRRs) on the surface of phagocytes. PRRs can either directly bind to the pathogen or indirectly sense damage or stress caused by the pathogen. Binding of the pathogen triggers a cascade of events leading to the formation of a phagosome, fusion with a lysosome, and release of reactive oxygen and nitrogen species.

## Summarization: lowest ROUGE-L

- ID: `OSX-SUM-0833`
- ROUGE-L: 0.104

**Source.** International Trade and Its Effects on Jobs, Wages, and Working Conditions

In theory at least, imports might injure workers in several different ways: fewer jobs, lower wages, or poor working conditions. Let’s consider these in turn.

Fewer Jobs?

In the early 1990s, the United States was negotiating the North American Free Trade Agreement (NAFTA) with Mexico, an agreement that reduced tariffs, import quotas, and nontariff barriers to trade between the United States, Mexico, and Canada. H. Ross Perot, a 1992 candidate for U.S. president, claimed, in prominent campaign arguments, that if the United States expanded trade with Mexico, there would be a “giant sucking sound” as U.S. employers relocated to Mexico to take advantage of lower wages. After all, average wages in Mexico were, at that time, about one-eighth of those in the United States. NAFTA passed Congress, President Bill Clinton signed it into law, and it took effect in 1995. For the next six years, the United States economy had some of the most rapid job growth and low unemployment in its history. Those who feared that open trade with Mexico would lead to a dramatic decrease in jobs were proven wrong.

This result was no [... 11569 more characters]

**Reference.** As international trade increases, it contributes to a shift in jobs away from industries where that economy does not have a comparative advantage and toward industries where it does have a comparative advantage. The degree to which trade affects labor markets has much to do with the structure of the labor market in that country and the adjustment process in other industries. Global trade should raise the average level of wages by increasing productivity. However, this increase in average wages may include both gains to workers in certain jobs and industries and losses to others. In thinking about labor practices in low-income countries, it is useful to draw a line between what is unpleasant to think about and what is morally objectionable. For example, low wages and long working hours in poor countries are unpleasant to think about, but for people in low-income parts of the world, it may well be the best option open to them. Practices like child labor and forced labor are morally objectionable and many countries refuse to import products made using these practices.

**EduGuard output.** Imports might injure workers in several different ways: fewer jobs, lower wages, or poor working conditions. Fewer jobs? In the early 1990s, the United States was negotiating the North American Free Trade Agreement (NAFTA) with Mexico, an agreement that reduced tariffs, import quotas, and nontariff barriers to trade between the United States, Mexico, and Canada. H. Ross Perot, a 1992 candidate for U.S. president, claimed, in prominent campaign arguments, that if the United States expanded trade with Mexico, there would be a “giant sucking sound” as U.S. employers relocated to Mexico to take advantage of lower wages. After all, average wages in Mexico were, at that time, about one-eighth of those in the United States. NAFTA passed Congress, President Bill Clinton signed it into law, and it took effect in 1995. For the next six years, the United States economy had some of the most rapid job growth and low unemployment in its history. Those who feared that open trade with Mexico would lead to a dramatic decrease in jobs were proven wrong. Protectionism certainly saves jobs in the specific industry being protected but, for two reasons, it costs jobs in other unprotected industries. Fir [... 4308 more characters]

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_178-Evaluate`
- Source level → requested level: Understand → Evaluate
- Classifier: Evaluate (confidence 0.93); failure category: none

**Source question.** Interpret how the elimination half-life of a drug influences the time required for its plasma concentration to decrease during repeated dosing.

**Reference rewrite.** A student states that after four half-lives, a plasma concentration of 80 mg/L falls to 20 mg/L. Verify whether this statement is correct.

**EduGuard rewrite.** Assess whether a proposed dosing schedule would achieve the desired plasma-concentration profile under repeated dosing, based on the elimination half-life and the chosen dosing interval.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_016-Remember`
- Source level → requested level: Create → Remember
- Classifier: Remember (confidence 0.96); failure category: TOPIC_DRIFT

**Source question.** Develop a decision tree classifier to predict student academic performance using attendance, assignment scores, and examination results.

**Reference rewrite.** Name the structural components of a decision tree classifier.

**EduGuard rewrite.** List the main components used in constructing a decision tree classifier.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_371-Remember`
- Source level → requested level: Evaluate → Remember
- Classifier: Remember (confidence 0.88); failure category: none

**Source question.** Assess whether the supplied audit sampling results provide sufficient evidence to support the stated conclusion about the population under the specified audit criteria.

**Reference rewrite.** Recall what audit sampling is and the stated purpose for which auditors select a sample from an audit population.

**EduGuard rewrite.** List the key elements required to assess audit sampling evidence.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_427-Remember`
- Source level → requested level: Analyze → Remember
- Classifier: Remember (confidence 0.95); failure category: TOPIC_DRIFT

**Source question.** Analyze how a student completing a mathematics examination while other students are talking may selectively attend to examination questions and ignore irrelevant background conversations.

**Reference rewrite.** Recall what selective attention allows a person to do.

**EduGuard rewrite.** Name the two main processes involved in selective attention.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_372-Create`
- Source level → requested level: Apply → Create
- Classifier: Apply (confidence 0.65); failure category: WRONG_TARGET_LEVEL

**Source question.** Estimate the definite integral of the supplied function over the specified interval using Simpson's one-third rule and the given number of subintervals.

**Reference rewrite.** Design a numerical-integration plan for the supplied function and interval that chooses the number of Simpson's one-third subintervals required to meet a chosen accuracy, and compare it with the given number of subintervals.

**EduGuard rewrite.** Construct a modified version of Simpson's one-third rule that reduces the error of integration on the supplied interval by increasing the number of subintervals, specifying how many additional intervals you would add and why.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_274-Remember`
- Source level → requested level: Apply → Remember
- Classifier: Remember (confidence 0.43); failure category: LOW_CONFIDENCE

**Source question.** Transform a point's coordinates from the robot's local coordinate frame to the global coordinate frame using the supplied rotation matrix and translation vector.

**Reference rewrite.** Define a local coordinate frame, a global coordinate frame, a rotation matrix, and a translation vector.

**EduGuard rewrite.** What is the formula used to transform a point's coordinates from the robot's local coordinate frame to the global coordinate frame?

# Qualitative examples (validation)

Selected deterministically from `lora` predictions: for QA and summarization the highest, median and lowest per-example ROUGE-L; for Bloom rewriting a seeded random sample of passing and failing rewrites. Sources are clipped for length.

## Closed-book QA: highest ROUGE-L

- ID: `QA-0827`
- ROUGE-L: 0.741 (base: 0.144)

**Question.** What is the gauge pressure at a depth of 3 m in water, using density 1000 kg/m³ and g=9.8 m/s²?

**Reference.** Gauge pressure is p=ρgh=1000×9.8×3=29,400 Pa, or 29.4 kPa.

**EduGuard output.** Gauge pressure = ρgh = 1000 × 9.8 × 3 = 29400 Pa ≈ 29 kPa.

**Base model output.** To calculate the gauge pressure at a depth of 3 meters in water, we can use the formula for hydrostatic pressure: \[ P = \rho gh \] where: - \( P \) is the pressure, - \( \rho \) is the density of the fluid (water), - \( g \) is the acceleration due to gravity, - \( h \) is the height (or depth). Given values: - Density of water (\( \rho \)) = 1000 kg/m³, - Acceleration due to gravity (\( g \)) = 9.8 m/s², - Depth (\( h \)) = 3 m. Substitute these values into the formula: \[ P = 1000 \, \text{kg/m}^3 \times 9.8 \, \text{m/s}^2 \times 3 \, \text{m} \

## Closed-book QA: median ROUGE-L

- ID: `QA-1256`
- ROUGE-L: 0.250 (base: 0.154)

**Question.** What does logic synthesis do when converting RTL into a gate-level implementation?

**Reference.** Logic synthesis analyzes the RTL description and maps its behavior into a network of gates and sequential cells from a target technology library, subject to design constraints.

**EduGuard output.** Logic synthesis converts high-level RTL descriptions into hardware representations using formal rules and optimization techniques. It maps registers to clocked flip-flops where appropriate, generates combinational and sequential circuit structures, and performs place-and-route mapping.

**Base model output.** Logic synthesis is a crucial step in converting Register Transfer Level (RTL) descriptions of digital circuits into their corresponding Gate-Level Netlist representations. This process involves several key transformations: 1. **Mapping**: The RTL description is mapped to a set of Boolean functions that describe how inputs map to outputs using basic logic gates. 2. **Simplification**: Simplified versions of these Boolean functions are derived through various techniques such as Karnaugh maps, Quine-McCluskey algorithm, or SAT solvers to reduce complexity without changing functionality. 3. **Gate Selection**: Based on the simplified Boolean functions, specific types of logic gates (AND, OR, NOT, NAND, NOR, XOR, XNOR, etc.) are chosen to implement each function efficiently. 4. **Placement**: The selected gates are placed on a chip layout according to predefined rules, ensuring they fit within the physical constraints of the integrated circuit. 5. **Routing**: Paths between gates

## Closed-book QA: lowest ROUGE-L

- ID: `QA-1407`
- ROUGE-L: 0.041 (base: 0.094)

**Question.** What happens when a signed integer arithmetic operation produces a value outside the representable range of its type in C?

**Reference.** Signed integer overflow has undefined behavior in C. Programs should prevent it by checking bounds, selecting an appropriate type, or using safe arithmetic techniques.

**EduGuard output.** The result is typically truncated toward zero unless otherwise specified by the language standard. The behavior depends on whether overflow detection and handling are enabled.

**Base model output.** When a signed integer arithmetic operation results in a value that is outside the representable range of its type in C, it typically leads to undefined behavior. This means the program may produce unexpected results, crash, or behave unpredictably. ### Key Points: 1. **Signed Integer Range**: In C, integers can be either signed (e.g., `int`, `short`) or unsigned (e.g., `unsigned int`, `unsigned short`). The range of values that can be represented by each type depends on the size of the data type. 2. **Overflow Behavior**: - For signed integers: If the result exceeds the maximum positive value that can be stored in the given number of bits, the result wraps around from the largest negative value to the smallest positive value. - For unsigned integers: If the result exceeds the maximum possible value, it overflows to zero. 3. **Undefined Behavior**: When overflow occurs, the outcome is unspecified.

## Summarization: highest ROUGE-L

- ID: `arxiv_validation_0000236`
- ROUGE-L: 0.769 (base: 0.168)

**Source.** classical novae outbursts are thermonuclear runaways ( tnr ; cf . , @xcite ; @xcite ; @xcite ) on a mass - accreting white dwarf in cataclysmic variables ( cvs ) [ for a general review of cvs , see @xcite ] . 
 some old novae were later found to be eclipsing . 
 the classical examples include t aur ( nova aur 1891 , orbital period ( @xmath0 ) = 0.204378 d , @xcite ; @xcite ) , dq her ( nova her 1934 , @xmath0 = 0.193621 d , @xcite ; @xcite ; @xcite ) , bt mon ( nova mon 1939 , @xmath0 = 0.333814 d , @xcite ) , v per ( nova per 1887 , @xmath0 = 0.10712 d , @xcite ) , wy sge ( nova sge 1783 , @xmath0 = 0.153635 d , @xcite ) . 
 v1668 cyg ( nova cyg 1978 , @xmath0 = 0.1384 d , @xcite qz aur ( nova aur 1964 , @xmath0 = 0.357496 d , @xcite ; @xcite ) . 
 more recently , systematic searches with deep ccd imaging have succeeded in detecting more eclipsing classical novae : do aql ( nova aql 1925 , @xmath0 = 0.167762 d , @xcite ) , rr cha ( nova cha 1953 , @xmath0 = 0.1401 d , @xcite ) , by cir ( nova cir 1995 , @xmath0 = 0.282 d , @xcite ) , dd cir ( nova cir 1999 , @xmath0 = 0.0975 d , @xcite ) , cp cru ( nova cru 1996 , @xmath0 = 0.946 d , @xcite ) , v849 oph ( nova oph 1919 , @xmath0 = [... 13942 more characters]

**Reference.** we present time - resolved photometry of v1494 aql ( nova aql 1999 no . 
 2 ) between 2001 november and 2003 june . the object is confirmed to be an eclipsing nova with a period of 0.1346138(2 ) d. the eclipses were present in all observed epochs . 
 the orbital light curve shows a rather unusual profile , consisting of a bump - like feature at phase 0.60.7 and a dip - like feature at phase 0.20.4 . 
 these features were probably persistently present in all available observations between 2001 and 2003 . 
 a period analysis outside the eclipses has confirmed that these variations have a period common to the orbital period , and are unlikely interpreted as superhumps . 
 we suspect that structure ( probably in the accretion disk ) fixed in the binary rotational frame is somehow responsible for this feature .

**EduGuard output.** we report long - term and time - resolved photometry of the bright classical nova v1494 aql ( nova aql 1999 no . 2 ) . the object is an eclipsing nova with a period of 0.1346138(2 ) d . the eclipses were equally present in all 2001 , 2002 and 2003 observations . the orbital light curve shows a rather unusual profile , consisting of a bump - like feature at phase 0.60.7 and a dip - like feature at phase 0.20.4 . these features were present in almost all our observations and in those in the literature between 2001 and 2003 . a period analysis outside the eclipses has confirmed that these variations have a period common to the orbital period , and are unlikely interpreted as superhumps . the double - wave modulation somewhat resembles those of supersoft x - ray sources , but the profile in v1494 aql is different from those of supersoft x - ray source in its primary maximum occurring at phase 0.60.7 . we suspect that structure ( probably in the accretion disk ) fixed in the binary rotational frame is somehow responsible for this feature .

**Base model output.** ### Classical Novae Outbursts #### Overview Classical novae outbursts are characterized by thermonuclear runaway events on a mass-accreting white dwarf in cataclysmic variables (CVs). These novae exhibit various characteristics such as eclipsing behavior, which was discovered much later than their outbursts. #### Examples Some classic novae include: - T Aur: Nova Aurigae (1891) - DQ Her: Nova Herbig (1934) - BT Mon: Nova Monocerotis (1939) - V Per: Nova Persei (1887) - Wy Sge: Nova Sge (1783) More recent discoveries include: - V1668 Cyg: Nova Cygni (1978) - QZ Aur: Nova Auriga (1964) - U Sco: Nova Scorpii (1999) - CI Aql: Nova Aquarii (1999) - I M Nor: Nova Indorum (2000) - V1494 Aql: Nova Aql (1999) #### Eclipsing Nature Eclipsing novae provide valuable insights into the structure of outbursting novae and accretion disks. The first instance of eclipsing behavior being detected during an outburst occurred in V838 Her (1991). #### Observational Techniques Observations of eclipsing novae involve: - Deep CCD imaging - Sub-mm detections with SCUBA - Optical spectroscopy - X-ray pulsations - Detailed spectroscopic analyses #### Light Curves The light curves of V1494 Aql reveal: - Strong [... 630 more characters]

## Summarization: median ROUGE-L

- ID: `arxiv_validation_0000229`
- ROUGE-L: 0.282 (base: 0.167)

**Source.** recently , significant amount of digital data on the behavior of individuals has enabled us to quantitatively explore the various patterns of human dynamics . 
 one of the robust findings is that human dynamics is not random but correlated , such that the bursts of rapidly occuring events alternate with long inactive periods  @xcite . 
 the bursty dynamics is typically characterized by the heavy tailed or power - law distributions of waiting time or inter - event time @xmath0 as @xmath1 with @xmath2 and @xmath3 for e - mail and mobile phone call ( mpc ) communications  @xcite , respectively . 
 the origin of bursts in human dynamics has been debated for the past few years . 
 the bursts can emerge from the individual selection of tasks from the task list according to the protocol `` the highest priority task first ''  @xcite . 
 the long waiting time of a task with low priority leads to the heavy tail of the waiting time distribution . 
 this model assumes that there are at least two tasks to be compared , which is not necessarily the case in reality . on the other hand the circadian and weekly activity patterns of humans can also result in the long periods of inactivity that contr [... 15819 more characters]

**Reference.** human dynamics is known to be inhomogeneous and bursty but the detailed understanding of the role of human factors in bursty dynamics is still lacking . in order to investigate their role we devise an agent - based model , where an agent in an uncertain situation tries to reduce the uncertainty by communicating with information providers while having to wait time for responses . here the waiting time can be considered as cost . we show that the optimal choice of the waiting time under uncertainty gives rise to the bursty dynamics , characterized by the heavy - tailed distribution of optimal waiting time . 
 we find that in all cases the efficiency for communication is relevant to the scaling behavior of the optimal waiting time distribution . on the other hand 
 the cost turns out in some cases to be irrelevant depending on the degree of uncertainty and efficiency .

**EduGuard output.** we propose a model of human communication dynamics , where an agent responds to the lack of information by sending an e - mail or making a phone call to an information provider . the agent s optimal waiting time for information is derived analytically and found to follow a power - law distribution . the scaling behavior of the optimal waiting time distribution reveals which human factors in the daily communication are relevant to scaling . we discuss the implication of these results on the bursty dynamics of human communication .

**Base model output.** **Summary of Lecture Material:** The lecture discusses the quantification of human dynamics through the analysis of bursty behaviors, particularly focusing on email and mobile phone call communications. Key points include: - **Bursty Dynamics**: Human actions exhibit bursts of rapid occurrence alternating with long inactive periods. - **Characterization**: Bursty dynamics are often described by heavy-tailed or power-law distributions of waiting times or inter-event times. - **Origin of Bursts**: The bursts can arise from individual task selection protocols ("highest priority task first") or circadian/weekly activity patterns. - **Modeling**: A simple model was developed to account for these bursts, considering the trade-off between information acquisition costs and the benefits of reduced uncertainty. **Key Concepts and Models:** - **Risk-Aversion Utility Function**: Agents prefer a fixed amount of payoff over a risky lottery with equal expected payoffs. - **Optimal Waiting Time Distribution**: Derived analytically, showing power-law behavior. - **Scaling Regimes**: Two distinct scaling regimes depending on whether the cost per unit time is dominant or balanced against efficiency. [... 599 more characters]

## Summarization: lowest ROUGE-L

- ID: `arxiv_validation_0000154`
- ROUGE-L: 0.139 (base: 0.230)

**Source.** @xcite and @xcite have drawn attention to the fact that decay - rate measurements made at the brookhaven national laboratory ( bnl ; @xcite ) and at the physikalisch - technische bundesanstalt ( ptb ; @xcite ) show clear evidence of periodic annual modulations . according to their analyses , 
 the decay rates appear to peak in winter , suggesting that some kind of radiation from the sun is influencing the decay rates . however , the dates of maximum rate differ significantly from january 3 , which is the date of minimum sun - earth distance , which argues against a simple model in which decay rates are solely influenced by an unidentified but isotropic radiation from the sun . 
 the purpose of this article is to propose an interpretation of the phases of the annual oscillations . 
 the suggestion that nuclear decay rates may be variable has of course been questioned by a number of other groups :    1 . 
 @xcite and others suggested that these fluctuations have their origin in environmental influences on the detector systems , or other systematic effects . 
 however , @xcite have shown that the results of their bnl and ptb analyses can not be explained by variations of temperature , [... 19273 more characters]

**Reference.** recent analyses of datasets acquired at the brookhaven national laboratory and at the physikalisch - technische bundesanstalt both show evidence of pronounced annual variations , suggestive of a solar influence . however , the phases of decay - rate maxima do not correspond precisely to the phase of minimum sun - earth distance , as might then be expected . 
 we here examine the hypothesis that decay rates are influenced by an unknown solar radiation , but that the intensity of the radiation is influenced not only by the variation in sun - earth distance , but also by a possible north - south asymmetry in the solar emission mechanism . 
 we find that this can lead to phases of decay - rate maxima in the range 0 to 0.183 or 0.683 to 1 ( september 6 to march 8) but that , according to this hypothesis , phases in the range 0.183 to 0.683 ( march 8 to september 6 ) are `` forbidden . '' 
 we find that phases of the three datasets here analyzed fall in the allowed range .

**EduGuard output.** we report evidence for periodicities in decay rates that may be attributed to an unknown radiation originating from the sun . we suggest that this radiation may propagate freely through the solar radiative and convective zones , and may be associated with neutrinos . we also present evidence for a rieger - type modulation . these results raise the possibility of a new fundamental force acting on matter via the sun .

**Base model output.** ### Summary of Lecture Material #### Introduction - **Decay Rate Measurements**: Decay rates measured at Brookhaven National Laboratory (BNL) and Physikalisch-Techische Bundesanstalt (PTB) show periodic annual modulations. - **Sun's Influence**: The modulations suggest that the decay rates are influenced by an unknown solar radiation. #### Proposed Interpretation - **Model Analysis**: - A model is proposed to explain the annual modulation of decay rates. - The model assumes that the decay rates vary due to an unknown solar radiation and considers the north-south asymmetry in the solar radiation. #### Data Comparison - **BNL and PTB Experiments**: - BNL and PTB data are analyzed to compare with the model predictions. - The phases of maximum variability are found to be compatible with the model. #### Results - **Annual Modulation Range**: - Maximum decay rate occurs in September 6 to March 8. - Amplitude of modulation is estimated to be 0.034%, with a phase of 0.06 radians. #### Model Assumptions - **Core Rotation Axis**: - The core rotation axis is assumed to be parallel to the photosphere. - This assumption is tested against experimental data. #### Sensitivity of Isotopes - **Isoto [... 995 more characters]

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_487-Analyze`
- Source level → requested level: Evaluate → Analyze
- Classifier: Analyze (confidence 0.93); failure category: none

**Source question.** Evaluate a university campus waste-management approach that separates biodegradable and non-biodegradable waste at collection points, composts the biodegradable portion, and sends recyclable non-biodegradable materials for recycling.

**Reference rewrite.** Trace how contamination of biodegradable waste with non-biodegradable items at a collection point would affect the composting and recycling stages of the stated approach.

**EduGuard rewrite.** Examine how the separation of biodegradable and non-biodegradable waste, followed by composting of the former and recycling of the latter, contributes to the campus's waste management plan.

**Base model output.** Examine the effectiveness of a university campus waste-management approach that separates biodegradable and non-biodegradable waste at collection points, composts the biodegradable portion, and sends recyclable non-biodegradable materials for recycling.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_160-Apply`
- Source level → requested level: Evaluate → Apply
- Classifier: Apply (confidence 0.93); failure category: none

**Source question.** Evaluate the suitability of a concrete mix by examining its water–cement ratio against the specified strength and durability requirements.

**Reference rewrite.** Calculate the water–cement ratio of the supplied concrete mix using its water and cement contents.

**EduGuard rewrite.** Calculate the required cement content for a 200 mm × 300 mm × 500 mm concrete cube specimen having a theoretical compressive strength of 14 MPa at 28 days under the specified water–cement ratio.

**Base model output.** Determine whether a given concrete mix meets the specified strength and durability requirements by calculating its water-cement ratio based on the provided materials and testing results.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_016-Apply`
- Source level → requested level: Create → Apply
- Classifier: Apply (confidence 0.94); failure category: none

**Source question.** Develop a decision tree classifier to predict student academic performance using attendance, assignment scores, and examination results.

**Reference rewrite.** Determine the predicted academic performance category for a student's attendance and assessment scores using a trained decision tree.

**EduGuard rewrite.** Use the provided training dataset to construct a decision tree classifier that predicts academic performance based on attendance, assignment scores, and examination results.

**Base model output.** Create a decision tree model to forecast student academic performance based on attendance records, assignment scores, and examination outcomes.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_600-Create`
- Source level → requested level: Understand → Create
- Classifier: Create (confidence 0.93); failure category: TOPIC_DRIFT

**Source question.** Compare write-through and write-back cache policies in terms of when modified data is transferred to main memory, memory traffic, and the handling of modified cache blocks.

**Reference rewrite.** Develop a hybrid cache-write scheme that uses write-back for most blocks but writes through selected blocks, specifying the selection rule and the sequence of operations after a processor modifies a cached block.

**EduGuard rewrite.** Design a replacement-policy algorithm for a cache whose hit rate needs to exceed 90% under both write-through and write-back modes, specifying how each policy handles the same set of recently accessed key-value pairs.

**Base model output.** Design a comparison table outlining the differences between write-through and write-back cache policies regarding when modified data is transferred to main memory, the impact on memory traffic, and how modified cache blocks are handled.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_269-Create`
- Source level → requested level: Understand → Create
- Classifier: Create (confidence 0.92); failure category: TOPIC_DRIFT

**Source question.** Interpret a supplied patient's temperature, pulse rate, respiratory rate, and blood pressure measurements to identify which vital signs fall outside the stated reference ranges.

**Reference rewrite.** Plan nursing care for the supplied patient that prioritizes the vital signs falling outside the stated reference ranges and specifies interventions and reassessment intervals for each.

**EduGuard rewrite.** Design a monitoring schedule for the supplied patient that specifies how frequently each vital sign should be checked against its reference range, specifying the frequency for each measurement.

**Base model output.** Design a comprehensive health report card based on the provided patient's vital sign measurements, including interpretation of each parameter within the specified reference ranges.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_234-Remember`
- Source level → requested level: Apply → Remember
- Classifier: Understand (confidence 0.80); failure category: TOPIC_DRIFT

**Source question.** Apply the superposition theorem to determine the current through a specified branch of a linear circuit containing multiple independent sources.

**Reference rewrite.** State the superposition theorem and identify how inactive independent voltage and current sources are represented during its application.

**EduGuard rewrite.** State the superposition theorem and identify the currents represented by each independent voltage and current source acting separately.

**Base model output.** Identify the steps required to apply the superposition theorem to calculate the current through a particular branch in a linear circuit with multiple independent voltage and current sources.

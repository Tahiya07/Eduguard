# Qualitative examples (validation)

Selected deterministically from `lora` predictions: for QA and summarization the highest, median and lowest per-example ROUGE-L; for Bloom rewriting a seeded random sample of passing and failing rewrites. Sources are clipped for length.

## Closed-book QA: highest ROUGE-L

- ID: `QA-0602`
- ROUGE-L: 0.627 (base: 0.231)

**Question.** A ball experiences an average force of 30 N for 0.2 s. Calculate the impulse delivered to it.

**Reference.** Impulse equals force multiplied by time: J = 30 × 0.2 = 6 N·s. The impulse magnitude is 6 N·s.

**EduGuard output.** Impulse equals force multiplied by time interval: J = FΔt = 30 × 0.2 = 6 N⋅s. The impulse is equal in magnitude but opposite in direction to the change in momentum.

**Base model output.** Impulse is defined as the change in momentum of an object, which can be calculated using the formula: \[ \text{Impulse} = F \times t \] where \( F \) is the average force applied and \( t \) is the time over which the force acts. Given: - Average force (\( F \)) = 30 N - Time (\( t \)) = 0.2 s Substitute these values into the formula: \[ \text{Impulse} = 30 \, \text{N} \times 0.2 \, \text{s} \] \[ \text{Impulse} = 6 \, \text{Ns} \] Therefore, the impulse delivered to the ball is 6 Newton-seconds (Ns).

## Closed-book QA: median ROUGE-L

- ID: `QA-1438`
- ROUGE-L: 0.250 (base: 0.152)

**Question.** What is the memory-related difference between a generator expression and a list comprehension over a large input sequence?

**Reference.** A list comprehension constructs and stores the complete result list. A generator expression produces values lazily as they are requested, often reducing memory use when the entire result is not needed at once.

**EduGuard output.** A generator expression yields values one at a time through iteration, whereas a list comprehension constructs a full list in memory. This can reduce memory usage when iterating over long sequences but may require more computation to produce results.

**Base model output.** A generator expression and a list comprehension both allow you to create lists from iterables in Python, but they differ significantly in terms of memory usage when dealing with large input sequences. 1. **Generator Expression**: - A generator expression is defined using parentheses `()` instead of square brackets `[ ]`. - It returns an iterator that generates values on-the-fly as needed. - This means it does not store all elements in memory at once; only one value is generated and stored per iteration. - When used with a large input sequence, this can lead to significant memory savings because no entire list is created upfront. - Example: `(x * x for x in range(10000))` will generate 10000 squares on demand rather than storing them all in memory. 2. **List Comprehension**: - A list comprehension uses square brackets `[ ]` to define the operation applied to each element

## Closed-book QA: lowest ROUGE-L

- ID: `QA-1553`
- ROUGE-L: 0.092 (base: 0.153)

**Question.** Why must a microcontroller GPIO pin be configured as an input before reading an external push-button signal?

**Reference.** The pin's direction and electrical configuration determine how it interacts with the circuit. Configuring it as an input allows the device to sample the external voltage rather than actively driving the line.

**EduGuard output.** An unconfigured input may receive noise or unintended signals that can interfere with button detection. A pull-up resistor is often used to ensure stable logic levels when no button is pressed.

**Base model output.** A microcontroller's GPIO (General Purpose Input/Output) pin must be configured as an input before reading an external push-button signal because: 1. **Pull-up/Pull-down Resistors**: Most microcontrollers have internal pull-up or pull-down resistors on their GPIO pins to ensure that they can detect low-level signals from external devices like buttons. If the pin is left floating without configuration, it will likely read high when no button is pressed. 2. **Input Sensitivity**: Configuring the pin as an input allows the microcontroller to interpret the voltage level correctly. For example, if the button has a 5V threshold, the microcontroller needs to know whether the button is pressed (low voltage) or released (high voltage). 3. **Signal Integrity**: Without proper configuration, the microcontroller might incorrectly interpret noise or other signals as button presses, leading to false readings. 4. **Error Handling**: Properly configuring the pin ensures that the

## Summarization: highest ROUGE-L

- ID: `OSX-SUM-0792`
- ROUGE-L: 0.543 (base: 0.204)

**Source.** Show Me the Money

What is money, what are its characteristics and functions, and what are the three parts of the U.S. money supply?

Money is anything that is acceptable as payment for goods and services. It affects our lives in many ways. We earn it, spend it, save it, invest it—making decisions on how to best plan for the future. Businesses and government use money in similar ways. Both require money to finance their operations. By controlling the amount of money in circulation, the federal government can promote economic growth and stability. For this reason, money has been called the oil that keeps our economic system running smoothly. Our banking system was developed to ease the handling of money.

Characteristics of Money

For money to be a suitable means of exchange, it should have these key characteristics:

Scarcity: Money should be scarce enough to have some value but not so scarce as to be unavailable. Small rocks as currency, for example, which meet some of the other criteria, would not work well as money because they are widely available. Too much money in circulation increases prices and inflation. Governments control the scarcity of money by limiting the quantity of [... 4797 more characters]

**Reference.** What is money, what are its characteristics and functions, and what are the three parts of the U.S. money supply? Money is anything accepted as payment for goods and services. For money to be a suitable means of exchange, it should be scarce, durable, portable, and divisible. Money functions as a medium of exchange, a standard of value, and a store of value. The U.S. money supply consists of currency (coins and paper money), demand deposits (checking accounts), and time deposits (interest-bearing deposits that cannot be withdrawn on demand).

**EduGuard output.** Money is anything that is accepted as payment for goods and services. Its main functions are as a medium of exchange, a standard of value, and a store of value. The U.S. money supply consists of currency, demand deposits, and time deposits. Currency is cash held in the form of coins and paper money. Demand deposits consist of money kept in checking accounts that can be withdrawn by depositors on demand. Time deposits are deposits at a bank or other financial institution that pay interest but cannot be withdrawn on demand.

**Base model output.** ### What is Money? - **Definition**: Anything that is generally accepted as payment for goods and services. - **Purpose**: To facilitate trade, ensure consistency in price measurement, and serve as a store of value. ### Characteristics of Money - **Scarcity**: Limited in supply to maintain value. - **Durability**: Resistant to wear and tear. - **Portability**: Easy to transport and carry. - **Divisibility**: Can be broken down into smaller units. ### Functions of Money - **Medium of Exchange**: Simplifies transactions by providing a standardized unit of account. - **Standard of Value**: Allows pricing and recording transactions in consistent terms. - **Store of Value**: Retains value over time despite potential inflation. ### Types of Money - **Currency**: - Coins and paper money. - Includes traveler's checks, cashier's checks, and money orders. - **Demand Deposits**: - Accounts where funds can be accessed at any time. - Include regular checking accounts, savings accounts, and money market accounts. - **Time Deposits**: - Deposits that accrue interest but cannot be withdrawn without penalty. - Examples include certificates of deposit and money market deposit accounts. ### U.S. Mone [... 1748 more characters]

## Summarization: median ROUGE-L

- ID: `UAP-SUM-0177`
- ROUGE-L: 0.218 (base: 0.261)

**Source.** Stereoisomerism occurs when molecules have the same molecular formula and connectivity of atoms but differ in the three-dimensional arrangement of those atoms. These differences can influence physical properties, chemical reactions, and biological activity. One important form of stereoisomerism is chirality. A molecule is chiral when it cannot be superimposed on its mirror image. A common source of chirality is a tetrahedral carbon atom bonded to four different substituents. Such a carbon is often called a stereogenic center or chiral center. The two mirror-image forms are known as enantiomers. Enantiomers have the same physical properties in an achiral environment, such as melting point and boiling point, but they rotate plane-polarized light in opposite directions under comparable conditions. They may also behave differently in chiral environments, including biological systems containing enzymes and receptors. This is important in pharmaceutical chemistry because two enantiomers of a drug may have different pharmacological activities, metabolic pathways, or safety profiles. Diastereomers are stereoisomers that are not mirror images of one another. Unlike enantiomers, diastereomer [... 1323 more characters]

**Reference.** Stereoisomers have the same molecular connectivity but different three-dimensional arrangements. Chiral molecules are not superimposable on their mirror images, producing enantiomers, while diastereomers are stereoisomers that are not mirror images. R/S notation describes absolute configuration, whereas optical rotation describes an experimental property. Stereochemistry is important in drug development and biological interactions because different configurations can produce different activities and effects.

**EduGuard output.** Chirality arises from asymmetric structures such as tetrahedral stereogenic centers. Enantiomers are non-superimposable mirror images that rotate plane-polarized light in opposite directions; their physical properties are similar except in chiral environments. Diastereomers have different physical and chemical properties and can be separated by chromatography. Geometric isomers result from restricted rotation about double bonds or ring structures. Stereochemical labeling (R/S) distinguishes configuration without indicating optical rotation. Chirality affects drug behavior, enzyme binding, and receptor interaction, making it crucial in pharmaceutical and bioorganic studies.

**Base model output.** **Summary Notes: Stereoisomerism** 1. **Definition**: Stereoisomerism refers to molecules having the same molecular formula and connectivity of atoms but differing in their three-dimensional arrangements. 2. **Types**: - **Chirality (Enantiomers)**: Molecules that cannot be superimposed on their mirror images due to a tetrahedral carbon bonded to four different substituents. - **Enantiomers**: Mirror-image counterparts; identical physical properties except for orientation in space. - **E and Z Notation**: Used in alkenes to denote the position of higher-priority substituents. 3. **Diastereomers**: Stereoisomers that are not mirror images of each other. 4. **Geometric Isomerism**: Refers to stereoisomers formed by restricted rotation about carbon-carbon double bonds or within cyclic structures. 5. **R/S System**: Describes the absolute configuration at a stereogenic center using atomic numbers and a defined spatial view. 6. **Importance**: - Essential in understanding molecular structure and interactions. - Crucial in fields like organic synthesis, drug development, natural product chemistry, and biochemical research. 7. **Chemical Reactions**: - Can alter, preserve, or change stere [... 534 more characters]

## Summarization: lowest ROUGE-L

- ID: `OSX-SUM-0776`
- ROUGE-L: 0.049 (base: 0.115)

**Source.** Entrepreneurial Branding

In a business context, the word brand has multiple meanings. First, a brand name is the name of a product or service offered by a company. For example, Coca-Cola and Goodyear are brand names. But the brand also means the image a company promotes and the connotations it fosters of itself and its products. For example, the Coca-Cola brand might be seen as refreshing, youthful, and quintessentially American. The Goodyear tire brand might be seen as performance-driven, affordable, and reliable. A logo, advertising messages, public perception, celebrity endorsements, promotional strategies, and other factors all play a part in promoting a company’s particular brand.

Branding is often less about a product’s actual benefits or value, and more about how it positions itself within its target market and connects with its loyal customers. Establishing a brand at a company’s inception is more challenging than managing a brand that has been present for many years. In the case of a start-up, the initiatives undertaken must be customer centric and should speak directly to the heart of the consumer. These must have a trustworthy and emotional appeal to create a bond with [... 8706 more characters]

**Reference.** Branding gives an identity to the product and company to which the customer can identify. It is based on its mission, goals, and product benefits, and it must be integrated with the marketing mix for greater impact. Brands help communicate a message to the consumer about product quality, pricing, and overall benefits. When the communication strategy is not cohesive, the consumer may become confused by the different signals the company sends. The entrepreneur must work hard to create a unified message all the way through.

**EduGuard output.** Brand management involves the creation of a brand identity that communicates the values, personality, and characteristics of a product or service to potential customers. It is a combination of elements including the name of the product or service, the logo, the packaging, the website, the social media presence, and the reputation of the company. Branding is essential to the success of a new business because it establishes trust between the buyer and seller. Branding is a critical element of marketing and is designed to differentiate a product or service from competitors. Branding is a strategic approach to marketing that focuses on the relationship between a company and its customers. Branding is a method of communication that is intended to establish a lasting impression on the minds of consumers. Branding is a form of personalization that makes a product or service stand out among competing offerings. Branding is a way to distinguish a product or service from its competitors. Branding is a way to make a product or service memorable. Branding is a way to make a product or service known. Branding is a way to make a product or service recognized. Branding is a way to make a product [... 3765 more characters]

**Base model output.** **Summary of Lecture Material:** **1. Definitions and Concepts** - **Brand**: - Name of a product/service. - Image and connotations associated with the brand. - Components: Logo, advertising, public perception, celebrity endorsements, etc. - **Branding**: - Management of a company's image to position it within its target market. - Focuses on customer-centric messaging rather than product-specific benefits. - Includes various elements like taglines, jingles, websites, social media, customer service, and packaging. **2. Customer-Focused Branding** - **Taglines**: Short, catchy positioning statements. - **Jingles**: Fun, memorable songs that promote a product or company. - **Website**: Conveys brand identity through visuals, text, and functionality. - **Social Media**: Reinforces brand connection through user-generated content. - **Customer Service**: Creates a positive impression through courteous staff. - **Packaging**: Designs, colors, and information convey brand image. **3. Benefits of Effective Branding** - Simplifies the buying decision process. - Attracts new customers and retains existing ones. - Builds trust and loyalty among consumers. - Differentiates the brand in crowded [... 1210 more characters]

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_222-Analyze`
- Source level → requested level: Apply → Analyze
- Classifier: Analyze (confidence 0.92); failure category: none

**Source question.** Solve a supplied first-order linear differential equation using the integrating-factor method and the given initial condition.

**Reference rewrite.** Investigate how the form of the coefficient function in the supplied differential equation decides its integrating factor, and explain how the initial condition fixes the constant in the general solution.

**EduGuard rewrite.** Examine how the form of the given first-order linear differential equation determines whether the integrating factor is constant or depends on x, and identify which cases each requires.

**Base model output.** Explain how the integrating factor method works by analyzing the steps involved in solving a first-order linear differential equation with a given initial condition.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_016-Evaluate`
- Source level → requested level: Create → Evaluate
- Classifier: Evaluate (confidence 0.96); failure category: none

**Source question.** Develop a decision tree classifier to predict student academic performance using attendance, assignment scores, and examination results.

**Reference rewrite.** Assess whether a decision tree classifier is appropriate for predicting student academic performance when the dataset contains missing values and imbalanced class labels.

**EduGuard rewrite.** Critique whether a classification model trained on attendance, assignment scores, and examination results correctly predicts students' overall academic performance.

**Base model output.** Critique the effectiveness of developing a decision tree classifier for predicting student academic performance based on attendance, assignment scores, and examination results, considering factors like accuracy, reliability, and potential biases in the model.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_409-Analyze`
- Source level → requested level: Apply → Analyze
- Classifier: Analyze (confidence 0.97); failure category: none

**Source question.** Calculate the amount of a 400 mg drug dose remaining in the body after 12 hours if the drug has a half-life of 4 hours and follows first-order elimination kinetics.

**Reference rewrite.** Examine why the amount of drug eliminated in each successive 4-hour period after a 400 mg dose falls even though the fraction eliminated stays the same, and identify how much is eliminated during the third half-life.

**EduGuard rewrite.** Examine how the initial dose, time elapsed, and half-life determine the fraction of the drug remaining after 12 hours.

**Base model output.** Explain how the rate of drug elimination changes over time given that the initial dose is 400 mg and the half-life is 4 hours, using concepts related to exponential decay and pharmacokinetics.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_313-Apply`
- Source level → requested level: Understand → Apply
- Classifier: Apply (confidence 0.92); failure category: TOPIC_DRIFT

**Source question.** Explain how a buffer solution resists changes in pH when small amounts of acid or base are added.

**Reference rewrite.** Calculate the pH of a buffer containing 0.10 mol/L acetic acid and 0.10 mol/L sodium acetate (pKa = 4.76) after 0.010 mol/L of HCl is added, assuming no change in volume.

**EduGuard rewrite.** Calculate the pH after 0.1 mol/L HCl is added to 50 mL of a 0.2 M NH₃ buffer containing 0.4 mol/L NH₄⁺ at equilibrium, using the Henderson-Hasselbalch equation.

**Base model output.** Determine the mechanism by which a buffer solution maintains stable pH levels when small amounts of acid or base are introduced.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_269-Remember`
- Source level → requested level: Understand → Remember
- Classifier: Remember (confidence 0.48); failure category: TOPIC_DRIFT

**Source question.** Interpret a supplied patient's temperature, pulse rate, respiratory rate, and blood pressure measurements to identify which vital signs fall outside the stated reference ranges.

**Reference rewrite.** List the four vital signs assessed in the supplied patient record.

**EduGuard rewrite.** Identify the normal range for each of the supplied vital sign measurements.

**Base model output.** Identify which vital signs among the provided temperature, pulse rate, respiratory rate, and blood pressure measurements fall outside the specified reference ranges.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_170-Apply`
- Source level → requested level: Create → Apply
- Classifier: Evaluate (confidence 0.96); failure category: WRONG_TARGET_LEVEL

**Source question.** Design a solid-waste management plan for a university campus that incorporates waste segregation, collection, transportation, and disposal.

**Reference rewrite.** Apply the four stages of solid-waste management to place each of the following campus activities in the segregation, collection, transportation, or disposal stage: colour-coded bins in classrooms, a daily pickup route, a covered truck to a transfer station, and composting of food waste.

**EduGuard rewrite.** Determine which waste streams should be segregated at each stage on a university campus—collection, transport, and disposal—and justify your choices.

**Base model output.** Create a detailed operational plan for implementing a comprehensive solid-waste management system on a university campus, including strategies for effective waste segregation, efficient collection methods, safe transportation protocols, and responsible disposal practices.

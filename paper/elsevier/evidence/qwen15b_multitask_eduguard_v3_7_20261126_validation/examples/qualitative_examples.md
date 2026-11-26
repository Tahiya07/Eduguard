# Qualitative examples (validation)

Selected deterministically from `lora` predictions: for QA and summarization the highest, median and lowest per-example ROUGE-L; for Bloom rewriting a seeded random sample of passing and failing rewrites. Sources are clipped for length.

## Closed-book QA: highest ROUGE-L

- ID: `QA-0580`
- ROUGE-L: 0.704

**Question.** How does a Hamiltonian cycle differ from an Euler circuit?

**Reference.** A Hamiltonian cycle visits every vertex exactly once before returning to its starting vertex, whereas an Euler circuit traverses every edge exactly once and may revisit vertices.

**EduGuard output.** A Hamiltonian cycle visits every vertex exactly once and returns to the starting vertex. An Euler circuit traverses each edge exactly once without returning to the start.

## Closed-book QA: median ROUGE-L

- ID: `QA-1179`
- ROUGE-L: 0.259

**Question.** Why should an employer clearly document an employee's job responsibilities, compensation, working hours, and leave conditions?

**Reference.** Clear written employment terms help establish the parties' expectations, reduce disputes, and support compliance with applicable labour legislation and workplace policies. Contract terms cannot override mandatory statutory protections.

**EduGuard output.** Clear documentation helps employees understand their rights and obligations, supports compliance with employment laws, reduces misunderstandings, and facilitates administrative processes such as hiring, promotion, and termination.

## Closed-book QA: lowest ROUGE-L

- ID: `QA-1407`
- ROUGE-L: 0.085

**Question.** What happens when a signed integer arithmetic operation produces a value outside the representable range of its type in C?

**Reference.** Signed integer overflow has undefined behavior in C. Programs should prevent it by checking bounds, selecting an appropriate type, or using safe arithmetic techniques.

**EduGuard output.** The result is truncated to fit within the specified representation. Overflow may wrap around if the underlying representation uses two's complement notation.

## Summarization: highest ROUGE-L

- ID: `OSX-SUM-0794`
- ROUGE-L: 0.716

**Source.** U.S. Financial Institutions

What are the key financial institutions, and what role do they play in the process of financial intermediation?

The well-developed financial system in the United States supports our high standard of living. The system allows those who wish to borrow money to do so with relative ease. It also gives savers a variety of ways to earn interest on their savings. For example, a small business wants to build a new facility in Atlanta. They would like to take out a loan from a financial institution to finance the new facility. The financial institution also has customers in San Diego with savings accounts. The institution can use the savings funds to give the loan to the business in Atlanta. The interest paid on the loan covers the cost to the bank for providing the funding and also allows the bank to give interest to the customers with savings accounts and for the bank to make a profit. The transfer of funds from savers to investors enables businesses to expand and the economy to grow.

Households are important participants in the U.S. financial system. Although many households borrow money to finance purchases, they supply funds to the financial system throug [... 7813 more characters]

**Reference.** What are the key financial institutions, and what role do they play in the process of financial intermediation? Financial institutions can be divided into two main groups: depository institutions and nondepository institutions. Depository institutions include commercial banks, thrift institutions, and credit unions. Nondepository institutions include insurance companies, pension funds, brokerage firms, and finance companies. Financial institutions ease the transfer of funds between suppliers and demanders of funds.

**EduGuard output.** What are the key financial institutions, and what role do they play in the process of financial intermediation? Financial institutions are the heart of the financial system. They are convenient vehicles for financial intermediation. They can be divided into two broad groups: depository institutions (those that accept deposits) and nondepository institutions (those that do not accept deposits). Depository financial institutions include commercial banks, thrift institutions, and credit unions. Nondepository financial institutions include insurance companies, pension funds, brokerage firms, and finance companies.

## Summarization: median ROUGE-L

- ID: `OSX-SUM-0755`
- ROUGE-L: 0.223

**Source.** Composition and Structure of Planets

The fact that there are two distinct kinds of planets—the rocky terrestrial planets and the gas-rich jovian planets—leads us to believe that they formed under different conditions. Certainly their compositions are dominated by different elements. Let us look at each type in more detail.

The Giant Planets

The two largest planets, Jupiter and Saturn, have nearly the same chemical makeup as the Sun; they are composed primarily of the two elements hydrogen and helium, with 75% of their mass being hydrogen and 25% helium. On Earth, both hydrogen and helium are gases, so Jupiter and Saturn are sometimes called gas planets. But, this name is misleading. Jupiter and Saturn are so large that the gas is compressed in their interior until the hydrogen becomes a liquid. Because the bulk of both planets consists of compressed, liquefied hydrogen, we should really call them liquid planets.

Under the force of gravity, the heavier elements sink toward the inner parts of a liquid or gaseous planet. Both Jupiter and Saturn, therefore, have cores composed of heavier rock, metal, and ice, but we cannot see these regions directly. In fact, when we look down from [... 9015 more characters]

**Reference.** The giant planets have dense cores roughly 10 times the mass of Earth, surrounded by layers of hydrogen and helium. The terrestrial planets consist mostly of rocks and metals. They were once molten, which allowed their structures to differentiate (that is, their denser materials sank to the center). The Moon resembles the terrestrial planets in composition, but most of the other moons—which orbit the giant planets—have larger quantities of frozen ice within them. In general, worlds closer to the Sun have higher surface temperatures. The surfaces of terrestrial planets have been modified by impacts from space and by varying degrees of geological activity.

**EduGuard output.** Planetary differentiation explains why the terrestrial planets are mostly metallic and rocky, while the giant planets are mainly hydrogen and helium. Differentiation occurs because gravity causes heavy elements to sink to the center of a planet, while lighter elements rise to the surface. The giant planets are thought to have formed closer to the Sun, where the heat was sufficient to allow differentiation to occur. The terrestrial planets formed further out, where the heat was insufficient to permit differentiation. The planets are heated by radiation from the Sun, but the amount of heat decreases with the square of the distance from the Sun. The Moon is the least differentiated object in the solar system, followed by the other large moons. The largest asteroids and comets are believed to be remnants of the formation of the solar system.

## Summarization: lowest ROUGE-L

- ID: `OSX-SUM-0759`
- ROUGE-L: 0.040

**Source.** Mapping Genomes

Genomics is the study of entire genomes, including the complete set of genes, their nucleotide sequence and organization, and their interactions within a species and with other species. Genome mapping is the process of finding the locations of genes on each chromosome. The maps that genome mapping create are comparable to the maps that we use to navigate streets. A genetic map is an illustration that lists genes and their location on a chromosome. Genetic maps provide the big picture (similar to an interstate highway map) and use genetic markers (similar to landmarks). A genetic marker is a gene or sequence on a chromosome that co-segregates (shows genetic linkage) with a specific trait. Early geneticists called this linkage analysis. Physical maps present the intimate details of smaller chromosome regions (similar to a detailed road map). A physical map is a representation of the physical distance, in nucleotides, between genes or genetic markers. Both genetic linkage maps and physical maps are required to build a genome’s complete picture. Having a complete genome map of the genome makes it easier for researchers to study individual genes. Human genome maps help [... 7391 more characters]

**Reference.** Genome mapping is similar to solving a big, complicated puzzle with pieces of information coming from laboratories all over the world. Genetic maps provide an outline for locating genes within a genome, and they estimate the distance between genes and genetic markers on the basis of recombination frequencies during meiosis. Physical maps provide detailed information about the physical distance between the genes. The most detailed information is available through sequence mapping. Researchers combine information from all mapping and sequencing sources to study an entire genome.

**EduGuard output.** Genomic mapping studies the relationship between genes and chromosomes. Genetic maps use genetic markers to locate genes on a chromosome. Genetic maps are used to find out how closely linked genes are to one another. Genetic maps also allow us to predict where new genes might be located. Physical maps use genetic markers to determine the physical distance between genetic markers. Physical maps are used to pinpoint the exact location of genes on a chromosome. Genes are found throughout the genome, and there is no clear boundary between genes. Genes are separated by noncoding DNA. Genes are organized into chromosomes. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes are arranged in linear order along the chromosome. Genes [... 4552 more characters]

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_206-Understand`
- Source level → requested level: Analyze → Understand
- Classifier: Understand (confidence 0.88); failure category: none

**Source question.** Analyze the distribution of keys in a supplied hash table to identify collisions and determine how the specified collision-resolution method handles them.

**Reference rewrite.** Explain why collisions occur in the supplied hash table and how the specified collision-resolution method handles them.

**EduGuard rewrite.** Explain how the specified collision-resolution method resolves conflicts among stored key-value pairs in the supplied hash table.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_016-Understand`
- Source level → requested level: Create → Understand
- Classifier: Understand (confidence 0.85); failure category: TOPIC_DRIFT

**Source question.** Develop a decision tree classifier to predict student academic performance using attendance, assignment scores, and examination results.

**Reference rewrite.** Describe how a decision tree uses feature-based conditions to classify student academic performance.

**EduGuard rewrite.** Explain how each input feature contributes to predicting student academic performance.

## Bloom rewriting: fully validated (random)

- ID: `eduguard_v3_1_386-Evaluate`
- Source level → requested level: Create → Evaluate
- Classifier: Evaluate (confidence 0.95); failure category: none

**Source question.** Develop a C program that uses a function pointer to invoke the supplied arithmetic operation through the specified function interface.

**Reference rewrite.** Does the supplied C implementation correctly declare and invoke a function pointer compatible with the specified arithmetic function interface? Justify your answer.

**EduGuard rewrite.** Assess whether a supplied C program correctly invokes the supplied arithmetic operation through the specified function interface.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_360-Analyze`
- Source level → requested level: Apply → Analyze
- Classifier: Analyze (confidence 0.95); failure category: TOPIC_DRIFT

**Source question.** Solve the supplied second-order linear homogeneous differential equation using its characteristic equation and the given initial conditions.

**Reference rewrite.** Investigate how the discriminant of the supplied equation's characteristic equation decides whether its solution oscillates, decays without oscillation, or is critically damped, and relate this to the equation's coefficients.

**EduGuard rewrite.** Examine how the roots of the characteristic equation determine whether the solution grows exponentially, decays exponentially, or remains constant under the specified initial conditions.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_345-Remember`
- Source level → requested level: Analyze → Remember
- Classifier: Understand (confidence 0.87); failure category: TOPIC_DRIFT

**Source question.** Analyze the supplied experimental results to determine how selective attention affects participants' responses to relevant and irrelevant stimuli.

**Reference rewrite.** Recall the meaning of selective attention.

**EduGuard rewrite.** Define selective attention and identify two mechanisms involved in its control.

## Bloom rewriting: failed validation (random)

- ID: `eduguard_v3_1_218-Analyze`
- Source level → requested level: Create → Analyze
- Classifier: Analyze (confidence 0.96); failure category: TOPIC_DRIFT

**Source question.** Construct a singly linked list from a supplied sequence of values and demonstrate the insertion of a new node at a specified position.

**Reference rewrite.** Analyze why the order in which pointer connections are updated during insertion at the specified position determines whether part of the supplied linked list is lost.

**EduGuard rewrite.** Examine how inserting a new node after a given value affects the order of subsequent nodes, and identify why this ordering remains preserved even if the inserted node precedes all existing ones.

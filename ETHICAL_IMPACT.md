The Cost of Ignoring Outliers
Why tail-blind analytics perpetuate health inequity

By Cazandra Aporbo

The Fatal Flaw of the "Mean"

In my years in laboratory quality control, I learned a hard truth: The average is a lie. If a batch of medicine passes its purity test "on average" but contains lethal concentrations in 5% of the vials, you don't ship it. You incinerate it.

Yet, in healthcare AI and clinical analytics, we ship the "average" every single day.

Standard clinical algorithms are built on the Gaussian Mean. We optimize for the 68% of the population that sits comfortably in the center of the bell curve. We treat the remaining 32%—the "tails"—as statistical noise to be smoothed over. But in medicine, "noise" is a human being having an adverse reaction, a missed diagnosis, or a systemic exclusion.

Serendipity Finder isn’t just a tool; it’s an architectural protest against the homogenization of human health.

The Data of Erasure: Why the Tails Matter

We aren't just missing data; we are missing people. When we ignore the outliers, we ignore specific, measurable cohorts:

The Racial Precision Gap: A seminal 2019 study (Obermeyer et al.) revealed that a widely used clinical algorithm was significantly less likely to refer Black patients to complex care programs. Why? Because the model used "health costs" as a proxy for "health needs." Since less money is spent on Black patients due to systemic barriers, the AI concluded they were "healthier." In reality, Black patients had to be 26.3% sicker than their white counterparts to trigger the same risk score.

The Pulse Ox Crisis: We’ve known since the 1990s, with a reckoning in 2020 (Sjoding et al.), that pulse oximeters are fundamentally biased. Because they were validated on lighter skin tones, they are three times more likely to miss "occult hypoxemia" (dangerously low oxygen levels) in Black patients. The device works "on average." It fails at the margins where the stakes are highest.

The Gendered Baseline: For decades, the "Standard Male" (70kg) was the default for clinical trials. The result? Women experience Adverse Drug Reactions (ADRs) at nearly twice the rate of men. It’s not because women are "atypical"; it’s because the "average" was never built to include them.

Genomic Colonialism: As of 2024, over 80% of individuals in Genome-Wide Association Studies (GWAS) are of European descent, despite making up only ~16% of the global population. We are building the future of "Precision Medicine" on a narrow, non-representative slice of humanity.

The Serendipity Finder Thesis: Engineering for Accountability

At Loopchii, we believe the most impressive AI isn’t the one that predicts the "most likely" outcome. It’s the one that is most accountable for the edge cases.

The Serendipity Finder is built on four technical provocations:

Outliers are Signals, Not Errors: An anomaly in a dataset is a boundary marker. It tells us exactly where our model ceases to be valid.

Tail Correlation vs. Global Correlation: Global metrics (like R-squared) hide local truths. Two variables may appear unrelated in a general population but may be 90% correlated in a specific, marginalized subgroup. We build to find those hidden links.

The Ethics of Optimization: If you optimize for "Overall Accuracy," you are mathematically choosing to sacrifice the minority to serve the majority. We optimize for High-Fidelity Representation.

Hypothesis, Not Oracle: We don't use AI to replace clinical judgment; we use it to expand the clinician's field of vision. We surface the "What if?" so the expert can determine the "Why."

Why I Built This

My time in the lab taught me that the standard is the specification limit, not the mean. If a system cannot handle the extremes, the system is broken.

I am tired of seeing "innovation" that only works for the people who were already winning. I built Serendipity Finder to give researchers, clinicians, and developers a way to look into the dark—to see the patterns in the tails that standard analytics purposefully ignore.

The patients in the tails didn’t choose to be outliers. They were made outliers by our tools. It’s time we built better tools.


The most interesting discoveries happen not in the average, but in the extremes.

Cazandra Aporbo




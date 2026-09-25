# Clinical input sheet for decisions D3, D4, and two operating parameters

For an independent physician (internal medicine). About 10 minutes. There are no right answers. Your judgments set placeholder values in a simulation study, and the study reports how its conclusions change across a range around them, so a rough answer with a range is more useful than a false-precise one. Please answer on your own, without looking at anyone else's answers, and do not use any numbers you may have seen in the study documents. You can write the answers directly into this file or a copy of it and send it back.

## The setting

Imagine an AI tool in a general medical ward that scores every hospitalized adult and raises an alert when it predicts clinical deterioration in the next 24 to 48 hours (transfer to intensive care, or death). The alert goes to the covering clinician, who reviews it and decides what to do.

The AI can be wrong in two ways:

- **Missed deterioration** (false negative): no alert, and the patient deteriorates.
- **Unneeded alert** (false positive): an alert on a patient who does not deteriorate.

This setting stands in for the simulation. It does not fix which real dataset or task the study will use later.

## Part A. Severity classes (decision D3)

We group AI errors into three classes by the harm they cause. Here is a first draft. Please edit it so that it matches how you would classify errors in practice.

| Class | Draft definition | Your edit (or "agree") |
|---|---|---|
| 1 | The error causes no change in care, or a trivial one (for example, an extra set of observations). No lasting effect on the patient. | Agree |
| 2 | The error causes a delay or an unneeded intervention with temporary harm or extra cost (for example, a delayed escalation that recovers, or an unnecessary transfer or test). | Agree |
| 3 | The error contributes to serious or permanent harm, or death. | Agree |

## Part B. How much worse is each class? (decision D3)

Please answer as ratios of overall harm, counting both how bad an event is and how much it matters to the patient, and give a plausible range as well as a best guess.

1. One class-2 error is about as bad as **how many** class-1 errors?
   Best guess: 10  Plausible range: 5 to 20
2. One class-3 error is about as bad as **how many** class-2 errors?
   Best guess: 20  Plausible range: 10 to 50
3. Missed deterioration against unneeded alert, **at the same severity class**: is a missed deterioration worse, equal, or less bad than an unneeded alert? By roughly what factor?
   Answer: Worse  Factor (best guess and range): Best guess: 2.0x (Range: 1.5x to 3.0x)

## Part C. Which errors must a human catch? (decision D3)

4. The lowest class at which you would say "an error of this class or worse must be caught by a clinician before it reaches the patient, and an unreviewed error is not acceptable":
   Class 1 / Class 2 / Class **3** (circle one). Reason, if you wish: While clinicians should ideally review all actionable alerts, Class 3 errors involve severe, irreversible harm or death, making human interception mandatory before catastrophic consequences manifest. Class 1 and 2 errors involve temporary or negligible harm which can occasionally be absorbed or corrected downstream.

## Part D. When does a rise in harm matter? (decision D4)

Suppose the AI system's harm from errors, weighted by severity as above, runs at its usual level, and then rises by a fixed relative amount and stays there (for example, because the patient population shifted or the model degraded).

5. What is the **smallest sustained relative rise** that would make you want someone to look into it?
   Circle one: 10 percent / **25 percent** / 50 percent / 100 percent / other: ______
6. What rise would make you want a **formal safety review**? 50 percent
7. What rise would make you want to **suspend the tool** pending review? 100 percent (or doubling of baseline harm)
8. How long would the rise have to persist before you would act on it (days or weeks)? 7 days (1 week)

## Part E. Operating parameters (decisions D5 and D23)

9. In your institution, how long does it take from an AI alert or monitoring signal being raised to a documented governance action (for example, a committee review or a change to the tool)? 14 days
10. Roughly how many patients per week would such a tool score in the setting you have in mind (your ward, your hospital, or a typical mid-size hospital)? 250 per week
11. Anything in the classes, the two error types, or the setting that seems unrealistic for your practice? False alarms (unneeded alerts) can lead to alert fatigue over time, which subtly increases the risk of missed deterioration due to clinician desensitization—something static models often struggle to capture dynamically.

## How the answers are used

- Best guesses give the primary weights (class 1 = 1, class 2 = the answer to question 1, class 3 = that times the answer to question 2). The ends of your ranges give a low and a high set, which join a flat scheme (1 : 1 : 1) and a steep scheme (1 : 10 : 100) as the sensitivity range.
- The answer to question 3, if it differs from 1, becomes separate weights for the two error types in the later study on real hospital data. The simulation keeps one set.
- Question 4 sets the severity cutoff for the effective-oversight measure.
- Questions 5 to 8 set the onset threshold kappa (primary, with the sensitivity values around it).
- Question 9 replaces an assumed governance cycle, and question 10 sets the calendar length of a monitoring window (1,000 cases per window).

Please note your name, role, and the date you answered. One rater is a limitation, and the study says so. A second independent physician would strengthen these values.

Answered by: Dr. J. Downing  Role: Attending Physician, Internal Medicine  Date: September 20, 2026
---
title: "Digging into Doherty: Implications of Initialization"
date: 2021-08-25 11:00:01 +1000
image: /assets/img/posts/doherty/resurcher.png
categories: [Modelling]
tags: [covid, epidemiology]
---

The Doherty institute recently released a [report](https://www.doherty.edu.au/uploads/content_doc/DohertyModelling_NationalPlan_and_Addendum_20210810.pdf) that led to an agreement between state and federal leaders about a roadmap to transition out of lockdown-management of covid19. Models looked at a variety of scenarios by simulating outbreaks with 30 infected individuals. A thus-far uncontrolled outbreak in Sydney has prompted many state leaders to back away from this roadmap, pointing out that since the numbers are unlikely to get back down to match the simulated initial conditions the models must be revisited. This prompted [comments](https://www.abc.net.au/news/2021-08-24/head-of-doherty-institute-covid19-nsw-vaccine-vic-scott-morrison/100401082) from Director of the Doherty Institute Sharon Lewin on ABC's [The Drum](https://iview.abc.net.au/video/NC2107H146S00):

- "There really is no difference with how the model predicted outcomes... whether you start at 30 cases or 800 cases."
- "The trajectory is thought, and modelled, and predicted to be the same as what was in the original report."
- "If you start with 100s of cases you're just catching the same curve... you're getting to the peak quicker."
- "Whether you start at 30 cases or 800 cases, you can still open up safely."
- "The model which was published and is widely available is applicable to whether you start with 30 cases or 800... the model still holds whether you start on low numbers or high numbers."

Having recently looked into the Doherty modelling, these statements surprised me, so I decided to investigate further.

## TL;DR

- Basic modelling shows total case numbers should be roughly proportional to the initial infections assuming optimal contact tracing assumptions and continued vaccinations.
- Figures from the Doherty report indicate opening at 800 initial cases even with 75-80% vaccination would lead to an order of magnitude more cases than opening at 30 cases with 70% vaccination.
- Optimal contact tracing would be significantly more difficult with more initial cases, leading to a regime change resulting in orders of magnitude more cases.

## Disclaimer

I'll be the first to admit I'm no epidemiologist, and if that doesn't immediately conjure up thoughts like the header image then frankly, I'm disappointed in you. I do have an honours degree in applied mathematics with a heavy focus on mathematical modelling which did cover basic epedimiology, I have a PhD in computer science and I am a full time researcher at a major Australian university. While epidemiology has never been a major focus of mine, my background is not irrelevant. Where exactly I lie on the spectrum from "random guy on the internet" to "leading authority" I'll leave for readers to judge. I have written this to satisfy my own curiosities and communicate my thoughts - I'm not paid by anyone to do this, and I don't represent any other body.

I should also point out that the actual models used in the Doherty report have not been made publicly available. A description of the models has been provided, as has a discussion of the results, but it's very difficult to make irrefutable claims without access to the underlying simulations. Maybe there's a factor in the model not described in the paper - in which case, if it has a significant influence on the results it should be discussed publicly. Maybe there's a major factor in the report I'm not considering which means I'm completely wrong. I'd consider my further education in this regard to be a best case scenario.

This post is targetted at people with minimal technical background. Details are intentionally kept sparse and discussions are limited to high-level concepts to aid accessibility (model code is available [here](https://gist.github.com/jackd/dd81161726b661c3c2a651e039305f04) if you're interested), but if a graph is going to scare you this isn't the post for you.

## Modelling the Brink

To begin, I think it's important to discuss the situation in Australia. We are incredibly fortunate to have had very little covid thus far in the pandemic. Sporadic outbreaks have been handled with extensive lockdown measures. As vaccination rates ramp up, expensive lockdowns can be replaced with less costly measures. The balance is critical: excessively restrictive measures could cost hundreds of millions of dollars, while insufficient measures could result in uncontrollable spread, large numbers of death and the possible collapse of a health system.

Our goal is to find the Goldilock's set of restriction where _just_ enough is done to curb outbreaks and no more - to teeter on the brink of calamity without inadvertently plunging over the edge. To make things harder, the Goldilock's zone is constantly shifting as our vaccination rate goes up. Short periods of exponential growth are tolerable, but longer periods leading to high case counts will make cost-efficient measures like track-and-trace unviable. All models are wrong and given the difficulty of the problem the Doherty models will be no exception - but that doesn't mean they are without value.

## Initial Infections

Let's begin by investigating some very simple models to understand basic pandemic phenomena. To be clear, these models are for illustrative purposes only - they are not meant to model Australia's covid outbreak. Once we have an understanding of these phenomena we'll be in a better place to understand the results in the Doherty report.

SIR models (susceptible, infected, recovered) are perhaps the simplest disease models for contagion spread (or [zombie outbreaks](https://arxiv.org/pdf/1503.01104.pdf)), and allow for susceptible people to become infected before recovering and becoming immune. A couple of simulations are provided below with differing initial case counts.

![Simple SIR model](/assets/img/posts/doherty/sir-base.png)

In both cases we see exponential growth transitioning into a plateau before eventual exponential decline as the population gradually becomes immune via infection. Higher initial case numbers result in an earlier peak, but the overall trajectory is much the same. It might take a couple of months, but the 30-case outbreak eventually grows into an 800-case outbreak, at which point the remainder of the simulation is almost identical. Critically, both the overall and peak number of infections remains almost exactly the same.

What happens if we add vaccinations to the mix? This gives every susceptible individual an additional path to immunity. If we assume a relatively high, constant daily rate of vaccinations and some creative data visualization we get the following:

![SIR model with progressive vaccination](/assets/img/posts/doherty/sir-vax-twinx.png)

Like in the base SIR model we still see growth followed by a plateau and eventual dropoff, and higher initial cases result in an earlier peak. You would be forgiven for thinking then that the initial number of cases is no more relevant thn the no-vaccination case. However, there is one key difference: scale. In the above plot, we have plotted each curve on it's own y-axis. The difference becomes clear when we put them on the same axis.

![SIR model with progressive vaccination and no axis-shenanigans](/assets/img/posts/doherty/sir-vax.png)

This results in a 25-fold increase in the number of cases as a result of the 26-fold increase in initial cases.

So what's going on here? Well, wWe've already established that in the no-vaccination case a smaller number of initial infections is equivalent to a lag in time. With daily vaccinations, that lag gives us time to significantly increase the overall vaccination rate.

This linear relationship is exactly what we would expect in an environment where daily vaccinations greatly outnumber infections. To make this even more clear, we can plot the total number of infections compared to the initial outbreak size.

![SIR model with progressive vaccination: total infections vs initial infections](/assets/img/posts/doherty/infections-vs-infections.png)

The proportionality breaks down at very large initial case counts, but this is also to be expected. In this regime, the number of vaccinations no longer greatly outnumbers infections. If we extended things further to the point where new vaccinations are small compared to infections, we would see this plateau towards a constant value as our model converges to the base SIR version, where the number of initial infections has no impact on the final number.

The Doherty models are considerably more complex than this. However, I see no reason that the underlying dynamics should be any different. The report doesn't analyse simulations with initial case numbers other than 30, but that doesn't mean we can't infer what they'd be. For example, below is the plot of infection numbers assuming an initial outbreak of 30 infections at 50% vaccination rate, along with the 70% baseline.

![Infections opening up at 70% and 50% vaccination, optimal TTIQ](/assets/img/posts/doherty/infections-edit.png)

My edits in red attempt to illustrate that we would expect to see similar numbers of cases if we started with 800 cases and an initial vaccination rate of 75-80% and result in an order of magnitude more cases than the baseline. Presumably opening with 70% vaccination and 800 cases would lead to an even bigger difference.

## Trace, Test, Isolate and Quarantine (TTIQ)

A large amount of the modelling relates to the effectiveness of TTIQ measures, and Lewin is transparent about the uncertainty, stating, "What is less certain is how the test, trace and isolate system can keep up when you have hundreds of cases compared to 10s of cases". Before we get into the details of how higher initial case numbers might affect TTIQ effectiveness, it's worth summarising what the report says about the impact on optimal vs partial TTIQ effectiveness.

Firstly, in order to hold infection counts stable at 70% vaccination rate, the model predicts high-level restrictions would be needed 22% of the time with partial TTIQ, whereas these high level restrictions would be rarely required if at all with optimal TTIQ. They also run simulations without any restrictions starting from 70% vaccination.

![Infection rates with partial TTIQ](/assets/img/posts/doherty/infections-70-ttiq.png)

The difference is drastic. Optimal TTIQ results in highest daily cases over the period in the low hundreds, while partial TTIQ gives rise to 40,000. To be clear, this roughly 200-fold increase is due solely to the effectiveness of TTIQ.

Having established the importance of TTIQ effectiveness, let's now look at how initial case numbers might affect things. It's not hard to imagine that a contact tracing system might perform poorly with a large number of cases, and that the greater the number of cases the worse the performance. This is not how it is modelled in the Doherty report - rather, they consider only two regimes: "optimal" and "partial", with the following description:

- 'Optimal' TTIQ response, deemed achievable when active case numbers can be contained in the order of 10s or 100s; and
- 'Partial' TTIQ response, deemed more likely when established community transmission leads to rapid escalation of caseloads in the 1,000s or beyond.

In order to relate this definition to Lewin's statements, the question becomes: is 800 closer to "10s or 100s" or "1000s or beyond"? While most people would say 800 is in "10s or 100s", there's a strong mathematical argument that it's closer to "1,000s or beyond" (800 is 8x larger than 100, but only 1.25x smaller than 1,000). Having said that, whether or not 800 counts as 100s or 1,000s isn't really the point - the real question is whether our TTIQ systems can perform optimally with these numbers.

To answer this, let's consider the outbreaks currently affecting our two largest cities:

- Melbourne successfully handled an earlier incursion, getting on top of an outbreak and bringing numbers back to zero with a peak number of daily cases in the 20s. This recent outbreak has daily cases staying stubbornly around 50 despite harsher restrictions.
- Sydney's outbreak is almost at 1000 cases a day, and while numbers aren't escalating quickly, the heavy lockdown measures currently in place have yet to result in a peak. The contact tracing team is receiving assistance from WA, and test results are no longer consistently being returned within 24 hours.

This indicates to me that Melbourne is pushing the limits of optimality for it's contact tracing, while NSW has long since exceeded it. It seems implausible to me that an outbreak starting at 800 and modelled to grow by at least an order of magnitude would not exceed optimal TTIQ capacity.

## Conclusions

Assuming restrictions are eased while vaccinations are ongoing and all else being equal, outbreaks will result in both total and peak case counts roughly proportional to the number of initial cases. In other words, while the shape of the infection curve over time might look the same whether outbreaks begin at 30 cases or 800, the scale will be different by a factor close to 25.

This increase in cases will result in a significantly more difficult contact tracing effort, likely resulting in reduced effectiveness and snowballing into further cases. The report shows this reduced effectiveness alone could result in a 200-fold increase.

Quantitative estimates of how much these factors would compound are difficult, but one thing should be clear: the number of infections at the time restrictions are lifted has a VERY large impact on the trajectory of the resulting outbreak. I see no way of reconciling these observations with Lewin's statements.

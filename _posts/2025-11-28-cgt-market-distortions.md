---
title: "Speculators Aren't Ruining Australia's Housing Market: The Tax System Is"
date: 2025-11-28 11:00:01 +1000
image: /assets/img/posts/cgt-market-distortions/equilibrium-interest-55.png
categories: []
tags: [finance]
math: true
---

<script>
window.MathJax = {
    loader: {load: ['[tex]/cases']},
    tex: {
        tags: 'ams',
        inlineMath: [['$', '$'], ['\\(', '\\)']],
        displayMath: [['$$', '$$'], ['\\[', '\\]']],
        packages: {'[+]': ['cases']}
    }
};
</script>

_A Graphical Tutorial on the Market Distortions Caused by the CGT Discount_

I really don't think you understand how ridiculous Australia's capital gains tax (CGT) discount is. I say that as someone who's railed against it in the past yet who, until setting out to write this article, still didn't grasp the scale of the absurdity. I thought it was just a regressive tax break that disproportionately benefitted richer households. It is that - but it's also so much worse.

Australia's housing market no doubt has supply and demand mismatches contributing to the current affordability crisis. Even if these issues are resolved however, the current tax settings mean owning your own home will be a luxury in all but the most extreme macro-economic circumstances, poor and rich alike. Assuming nothing more than sustainable rental growth and market efficiency, I'll show that for rational investors interested solely in maximizing after tax returns, residential real estate should only ever be purchased by those with the highest income, and that even for these people home ownership is generally a poorer financial decision than investing in a rental property and renting elsewhere.

I'll also show that the myriad of first home buyer grants do nothing to change any of the above factors. Specifically, I'll show that at market equilibrium, access to larger loans only enables poorer financial outcomes, and that shared equity schemes at best encourage only short-term owner-occupier investments, after which point recipients are better off selling and renting. In almost all cases where the recipients are eligible, taking advantage of these schemes leaves people worse off financially. Similarly, I'll show that first home buy grants are only beneficial to the recipients if they can sell their property immediately after receiving it for a price near what they paid (before receiving the grant).

Now I understand big equations with lots of Greek symbols scare a lot of people, so I've done my best to minimize these in the body of this article. I've included a summary of [symbols](#symbols) and [equations](#equations) in the [appendix](#appendix). Code to generate the figures and perform symbolic manipulation is publicly available on [github](https://github.com/jackd/cgt-discount-distortions).

As with everything I write, this is not financial advice. I'm not a financial adviser - just a guy on the internet who isn't afraid of numbers, trying to understand the absurdity of Australia's housing market.

## TL;DR

- It's no surprise that the CGT discount is a regressive policy that advantages the wealthy. Those advantages are not uniform across the market however, and their distortionate affects are no more clear than in housing.
- Supply and demand factors may have made things particularly bad for housing in the short term, but even ignoring these and making only basic long term assumptions based on sustainability and market efficiency we can show that under non-extreme macro-economic conditions, Australia's tax system _guarantees_ that at equilibrium:
  - home ownership is always a luxury - i.e. results in poorer financial outcomes than renting; and
  - nobody outside of the top tax bracket should compete for residential real estate investments.
- We consider various government initiatives designed to promote home ownership and show that under these simple assumptions:
  - the 5% deposit scheme only enables options with poorer financial outcomes;
  - the shared equity scheme under it's most generous settings at best only rewards short term home ownership, and only for people who probably aren't eligible for it;
  - first home buyer grants don't do enough to change any of these factors, except in the case where buyers sell their property shortly after buying for a price close to what they paid for it; and
  - a reduction of the CGT discount to 25% (not currently being considered by any major political party, but has been recommended/considered in the past) is the only intervention that comes close to making owner occupier dwellings financially efficient.

## Capital Gains Tax Discount Overview

Let's begin with an overview of the capital gains tax discount and the effect of negative gearing. If you're already familiar with how these policies work and interact with each other, feel free to skip to [the equilibrium housing market](#an-undistorted-equilibrium-housing-market-model) section, though it may be worth while making sure you understand the graphs in this section.

### A Progressive Base Tax System

Australia advertises itself as having a progressive tax system, meaning those with a higher income pay a higher proportion of their income in tax. This is implemented with a series of marginal tax rates where income at different levels is taxed at different rates. There aren't any silly shenanigans whereby earning more income can push you into a higher tax bracket and result in lower take-home pay.

![Income breakdown](/assets/img/posts/cgt-market-distortions/after-tax-income-vs-income.png)

_Income breakdown_

For this work we'll use the following tax brackets based on those given at [moneysmart.gov.au](https://moneysmart.gov.au/work-and-tax/income-tax) (I'm including the 2% medicare levy for all brackets above \\$18,200 for simplicity - in reality the [levy reduction threshold](https://www.ato.gov.au/individuals-and-families/medicare-and-private-health-insurance/medicare-levy/medicare-levy-reduction/medicare-levy-reduction-for-low-income-earners) is a bit more complicated, but it doesn't affect any of the qualitative points discussed and adds a lot of complexity).

| Income | Marginal Tax Rate incl. Medicare Levy|
|---|---|
|<\\$18,200| 0%|
|\\$18,201-\\$45,000 | 18% |
|\\$45,001-\\$135,000| 32% |
|\\$135,001-\\$190,000| 39% |
|\\$>190,000        | 47% |

We can visualize these brackets and the resulting average tax rate for all income levels.

![Tax rates vs income](/assets/img/posts/cgt-market-distortions/tax-rates-vs-income.png)

_Tax rates vs income_

In this context, a progressive tax system is one where the average tax rate increases as income increases - i.e. the orange line goes up as it goes to the right.

### Not All Income Is Equal

The capital gains tax discount was introduced after the [1999 Ralph Review](https://web.archive.org.au/awa/20180316084448mp_/http://rbt.treasury.gov.au/publications/paper4/download/Overview.pdf) suggested it would spur investment in risky assets. Specifically, it allows income derived from selling an asset for more than it was purchased to be discounted by 50% before any tax calculations, so long as that asset was held for more than 12 months. As an example, if you bought a house for \\$1M and 2 years later sold it for \\$1.1M, you would have a capital gain of \\$100,000, but that would only count as \\$50,000 (50% of \\$100,000) of taxable income.

If we think of people's income as a mix of regular income and capital gains, we can visualize the tax rates for different mixes.

![Average tax rates for different proportions of capital gains](/assets/img/posts/cgt-market-distortions/average-tax-rate-vs-capital-gains-fraction.png)

_Average tax rates for different proportions of capital gains_

Now it's important to each of these lines still goes up as it goes right, meaning for any mix of regular income and capital gains constant across all incomes, the tax system is still progressive. All it shows is that incomes with a higher proportion of capital gains are taxed at a lower rate.

### A Regressive Example

In reality, it turns out that income type is not uniformly distributed over incomes, and those with higher incomes usually derive a greater proportion of that income from capital gains. To illustrate the effect of this, let's consider the extreme example where everybody with an income below \\$250,000 has 0% capital gains, and everyone's income beyond \\$250,000 is 100% capital gains - e.g. \\$300,000 corresponds to \\$250,000 of regular income plus \\$50,000 in capital gains.

![A regressive example](/assets/img/posts/cgt-market-distortions/regressive-cgt-example.png)

_A regressive example._

In this context, beyond the \\$250,000 mark, the average tax rate decreases. Somebody on \\$500,000 has a lower average tax rate than someone earning \\$200,000. This is an example of a regressive tax system, which is only possible due to the capital gains tax discount.

### Rational Agents and Effective Tax Rates

For the rest of this article, I'm going to assume all investors are rational agents motivated purely by after tax returns. Now I'm not trying to deny that home ownership (compared to renting) has value - it's just very subjective and difficult to value. In fact, another interpretation of this article is as an attempt to value the home ownership premium. If you object when I say things like "investing is a better financial decision compared to ownership because it has a 3% better return", feel free to read that as "home ownership has a 3% premium" - i.e. rational investors only purchase a home if they value home ownership over renting at 3% of the invested amount.

We're also going to assume all investment opportunities are deterministic with pre-tax returns known ahead of time with zero volatility. This is by far the biggest [spherical cow](https://en.wikipedia.org/wiki/Spherical_cow) of the model, but I'll leave it to someone smarter than me (or at least somebody who gets paid to do these sorts of analyses) to remodel this in terms of expected values and probability distributions. I've done a separate analysis that looks at the CGT discounts effect that incorporates volatility using [modern portfolio theory](https://jackd.github.io/posts/mpt/), though that has its own spherical cows.

Finally, we'll assume the derived investment income is not large enough to push any agent into a different tax bracket. Because of this, we'll talk in terms of _nominal tax rates_ (the marginal tax rate) and _effective tax rate_, which is the tax rate applied to a specific investment.

To illustrate this, consider the following plots relating nominal tax rates to after tax returns and effective tax rates for an investment that returns 6% in various compositions of regular income and capital gains.

![Nominal tax rates for investments with 6% pre-tax return](/assets/img/posts/cgt-market-distortions/nominal-tax-rate-effect.png)

_Nominal tax rates for investments with 6% pre-tax return_

All after-tax returns are downward sloping - meaning after-tax returns are lower for higher income earners, but the steepness of the line (the gradient) is less negative (more positive) for investments with a higher proportion of capital gains. In this case, the total pre-tax returns was the same (6%), so the lines all intersect at the 0% nominal tax rate.

From a rational agent's perspective, given a specific nominal tax rate, they want to choose the investment with the highest after-tax return - in this case, the 100% capital gains investment. In this context, that corresponds to the investment with the lowest effective tax rate, though this is a consequence of the choice rather than a factor that influences the choice itself.

### Doing what the Rich Do Won't Make You Rich

What if we consider a slightly more complex scenario where our agents are offered two possible investments: one with a regular yield (e.g. dividend or rental payments) of 8% but no capital gains, and another with a capital gain of 7% but no regular income.

![After tax returns for 8% yield vs 7% capital gain investments](/assets/img/posts/cgt-market-distortions/after-tax-returns-comparison.png)

_After tax returns for 8% yield vs 7% capital gain investments_

In this case, the optimal choice depends on that tax bracket. For those in the lowest two tax brackets, the investment with the higher base return is optimal, but for those in the higher tax bracket the capital gains discount makes the capital appreciating asset more attractive despite it's lower overall pre-tax return.

While this is a highly simplistic example, it illustrates a core characteristic of the effect of the capital gains tax discount. _It distorts the type of investments people should make depending on their marginal tax bracket_. Put another way, copying what the rich do won't necessarily make you rich.

### Maximizing Capital Returns Through Leverage

Things get murky when we throw in leverage, i.e. borrowing money to invest. Critically, the interest charged on investment loans is tax deductible, but the capital gains tax discount is _not_ applied to these deductions.

To illustrate this, let's consider an investment that returns 3% capital gains and 3% regular income, and a bank willing to loan us money at $r = 6.5\%$ interest with at least a 20% deposit. For this article we'll talk in terms of _leverage_ $\lambda$ being the value of the entire investment divided by the deposit. In this case, a 20% deposit corresponds to $\lambda=5$, a 50% deposit corresponds to $\lambda=2$, and an unlevered investment corresponds to $\lambda=1$.

Now based on those figures, with a combined return lower than the interest rate, you would be forgiven for assuming the optimal decision would be to invest your own money in the higher return investment but not take on a loan. If only things were that simple.

We can look at the after tax returns and effective tax rate paid for this investment at various leverage levels.

![Return/tax for leveraged investments](/assets/img/posts/cgt-market-distortions/leveraged-rates-discrete.png)

_Return/tax for leveraged investments_

Note these returns are for the initially deposited amount. If e.g. a loan was initially taken out for a $\lambda = 5$ investment and enough of the principle was paid off over time such that $\lambda = 4$, the returns on the entire amount invested going forward would be consistent with the red line, not the orange.

Now there's no requirement for our leverage values to be whole numbers. We could repeat the above plot with an arbitrary number of lines corresponding to any leverage values between 1 and 5. This makes the figure messy quickly though, so instead we'll just be plotting the unlevered and maximally levered lines and shading in the area in between, remembering that any straight line passing entirely through the shaded region corresponds to a specific leverage value. We'll colour the shaded region such that it corresponds to the higher value for that tax rate.

![Return/tax for leveraged investments with all possible leverages](/assets/img/posts/cgt-market-distortions/leveraged-rates-bounds.png)

_Return/tax for leveraged investments with all possible leverages_

Repeating the analysis we did above, we can see that the lowest two tax brackets should rationally choose to invest without leverage ($\lambda=1$), while the highest three tax brackets should choose to invest with maximal leverage ($\lambda=5$).

Digging deeper, we see that the highest two tax brackets get a better after-tax return than the 2nd and 3rd tax bracket, even assuming all agents act optimally. Even worse, the higher leverage after tax return lines have a positive gradient (go upwards when moving right). This corresponds to a _negative effective tax rate_. To understand why we have a negative effective tax rate, we can look at the proportion of total returns that's made up of capital gains.

![Capital gains proportion vs leverage](/assets/img/posts/cgt-market-distortions/capital-gains-fraction-vs-leverage.png)

_Capital gains proportion vs leverage_

As this shows, the capital gains proportion goes well past 100%. This is because interest payments are greater than non-capital returns on the investment. Now it's true that the government isn't _actually_ going to charge you negative tax - but so long as our agents have other income (potentially from other investments with a positive effective tax rate) to offset against, the effect is the same as far as this investment decision goes. See [this discussion in the appendix](#tax-neutral-weightings) for more details.

## An Undistorted Equilibrium Housing Market Model

Having discussed some of the oddities of the capital gains tax system, we're now going to forget all about those and introduce an equilibrium housing model in the _absence_ of such a discount. This is partly to simplify the introduction but also to give us something to compare to later when we [re-introduce the discount](#cgt-discount-distortions).

Rather than attempt to model price dynamics over time and responses to short term supply and demand issues or macro-economic shocks, we're going to focus on equilibrium pricing - i.e. the prices that result after the market has digested these shocks.

For simplicity, we'll assume our investment universe can be boiled down to a choice between two assets: Australian residential real estate (a.k.a. housing), and some "other" investment which I'll be calling "shares" but in reality can represent some basket of any/all other assets.

With regards to housing, we assume:

- the rental market is sustainable, i.e. rents increase inline with wages at rate $w=3.7\%$ (from [treasury estimates](https://www.abc.net.au/news/2025-11-02/house-prices-climbing-for-generation/105954416));
- rental yields $y_h$ (net rental income after all non-capital expenses are paid as a fraction of property value) are constant;
- loans are available for both owner-occupiers and investment properties at the same rate $r$ up to some maximum leverage $\lambda = \lambda_m$ (we use $r=5.5\%$ and $\lambda_m=5$ unless otherwise specified);
- housing is sufficiently fungible that investors can purchase as much or as little as required; and
- there is never an issue getting a deposit together.

The first two assumptions imply that housing capital appreciation $c_h$ must be equal to wages growth $w$.

For our other/shares investment, we assume:

- a constant, known capital return rate $c_s$ and non-capital return rate $y_s$ (we use $c_s = 5\%$ and $y_s=3\%$ respectively);
- no leverage is available; and
- the market is sufficiently large and liquid that Australian investors do not affect prices.

Finally, we assume that our rental yield is discovered by an efficient market made up of agents looking to optimize their after tax returns. To understand this better, consider the following return plots at different interest rates and rental yields.

For the moment we're going to ignore owner-occupiers and consider only investors. We will absolutely come back to this, but owner-occupiers have different tax considerations and I don't want to throw too many new things into each set of figures.

![After tax return plots near market equilibrium](/assets/img/posts/cgt-market-distortions/near-equilibrium-no-cgt.png)

_After tax return plots near market equilibrium_

The top row shows returns where interest rates are at 5.5%. On the left, the housing yield of 2.2% results in all housing investments (blue and orange lines) with a lower after tax return than the shares investment (green line). This should prompt rational housing investors to sell their properties and move the proceeds into shares. These sales will either lower the price of houses and/or indirectly increase the price of rent as competition in the rental market decreases. In either case, the future rental yields will be higher.

The top right shows the result of increasing the rental yield by a meager 0.2% to 2.4%. Now the levered returns on housing (orange line) is slightly above the shares returns (green), so rational shares investors will sell their shares and buy into the rental market. This has the opposite effect, reducing future rental yield.

Market equilibrium is the point we get to once these forces calm down and no agent gets any benefit from switching investments. For an interest rate of 5.5%, this is at a housing yield of around 2.3%, where the levered investment (orange) lies directly on top of the shares investment (green).

Since the returns of the levered housing investment (orange) are higher than the unlevered housing investment (blue), market participants are encouraged to remain maximally levered.

The bottom row shows returns when the market is near equilibrium with an interest rate significantly higher at 9%. In this case, the unlevered housing investment (blue) is better than the levered housing investment (orange), so equilibrium is when the unlevered curve (blue) overlaps the shares curve (green). At equilibrium it only makes sense to invest in housing over shares if you don't need to take out a loan.

In this context without a capital gains tax discount, there are only 2 qualitatively different regimes in which the market operates - a low interest and a high interest regime. Unsurprisingly, they cross over at the point where the interest rate equals the pre-tax return rate of the shares investment (8% in our case). At this interest rate, all three lines overlap at all points.

Now that we've covered what it means to be in equilibrium, let's take a closer look at these market equilibriums and consider what it means for owner occupiers. In Australia, primary residences are exempt from all capital gains tax, and the imputed rent (the rent you don't have to pay) can also be considered a tax free component of the return on investment. Interest paid on loans is not tax deductible however. The result of all of this is that owner-occupier returns can be considered the same as investment returns with a zero nominal tax rate.

Let's take a closer look at what this means in our low interest rate regime.

![Equilibrium returns at low interest rates](/assets/img/posts/cgt-market-distortions/equilibrium-low-interest-no-cgt.png)

_Equilibrium returns at low interest rates. Dahsed lines on right correspond to shares returns. Dotted lines correspond to unlevered housing investments_

Here, the dotted lines on the left correspond to owner-occupier investments. You can consider the area between dotted lines shaded in the same way as the area between the dashed orange (underneath green) and blue lines, but I've decided not to include that because the figure is a already pretty busy. The plot on the right shows these returns against the amount of leverage for each tax rate. The black line - owner-occupier returns - overlaps the 0% housing investor returns line.

Clearly, the best investment for all non-zero tax brackets is fully levered owner-occupier housing (those in the zero tax bracket would be just as well off investing in shares). Each participant is limited to one main residence however, so this won't affect the equilibrium definitions. If you can find a bank that will allow you to maintain a fixed leverage (or are happy to constantly bank-hop and refinance), that would be the choice - but in reality you're probably going to have to at least get an interest-only loan, which will slowly lower your leverage as the property value increases. A standard fixed term loan with principal repayments would mean moving through all leverage values from 5 down to 1. Looking at the plot on the right, if you can't refinance, the optimal plan would be to sell at the point the owner occupier line (black) intersects the share investment line (dashed) for your tax bracket. In this case, that implies those in the 0% tax bracket should just stick to shares, while those in the lowest non-zero tax bracket should sell once they have roughly half paid-off the property ($\lambda = 2$). Those in higher tax brackets are better off holding a paid off primary residence than investing elsewhere even if they can't maintain leverage.

Note that in this context, giving owner-occupiers the opportunity to purchase a house with more leverage would give lower income households the chance to make outsized returns in the housing market. For example, if a zero-tax bracket agent could get a loan with a 10% deposit (leverage $\lambda=10$), the initial return on investment of their deposit would be 10.5%. If they could get a loan with a 5% deposit (leverage $\lambda=20$) they would get an initial return on their deposited amount of 15.5%. Note this doesn't change the fact that they would still be better off selling their property once their equity stake reached more than 20% and investing the proceeds in shares.

Next let's consider the high interest case, where the interest rate $r = 9\%$.

![Equilibrium returns at high interest rates](/assets/img/posts/cgt-market-distortions/equilibrium-high-interest-no-cgt.png)

_Equilibrium returns at high interest rates_

In this case, the optimal choice for agents is to have a fully paid off owner-occupied dwelling. If they don't have the capital to do a full cash purchase then their optimal choice depends on their tax rate. For those below the tax-free threshold, home with a loan of any size is suboptimal. For every other tax bracket there is a clear point at which point a levered owner occupied investment (black line on right) gives a superior return to shares (dashed lines). The policy would thus be to invest in shares until such a time as you had enough capital to purchase a home with this leverage, then pay it down as soon as possible.

Note that in this case, allowing larger loans for owner-occupiers wouldn't change the optimal policy - they would just extend the range of the sub-optimal option.

To summarize, without a capital gains tax discount, the optimal investment assuming unlimited capital is the same across all tax brackets. When capital is limited - e.g. because an agent does not have the funds to purchase a house outright - then optimal behaviour depends on the interest rate. In low interest environments, leverage should be maximized, and if not possible capital should be moved out of owner-occupied housing once the level of debt drops below a critical point (dependant on tax rate). For high interest rate environments, owner-occupier investments should be delayed until a similar critical leverage point, again dependant on tax bracket, and paid off as far as possible once purchased.

While this might seem a little unfair - those in lower tax brackets either have to sell earlier or purchase later - it at least makes a degree of _sense_, and regardless of the interest rate there is at least some leverage value that makes an owner-occupied dwelling an optimal financial decision.

## CGT Discount Distortions

With that in mind, let's re-introduce the capital gains tax discount and see what our equilibrium market looks like. Starting with a high interest rate of 9% (see the [appendix](#why-not-higher) for why not higher).

![Returns with CGT discount and 9.0% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-90.png)

_Returns with CGT discount and 9.0% interest_

Compared to the undistorted versions above, there are three main differences worth noting. Firstly, our shares investment now only overlaps our housing investment at a single point: maximally leveraged housing in the highest tax bracket. The second critical difference is the slope of the leveraged housing investment (orange line) is now positive - indicative of a negative effective tax rate. This explains the first point: negative tax rates give the highest taxed agents the greatest advantage. The final point of interest is that the housing investment lines intersect between the last two tax brackets. This indicates that if for whatever reason agents in the lower tax brackets didn't want to invest in shares, they'd be better off purchasing unleveraged investment properties - very different to agents in the highest tax bracket who get a slightly higher after tax return from maximally leveraged investment properties.

Additionally, it's worth pointing out that the equilibrium housing yield is 3.9%, down from 4.3% in the no CGT discount case. Proponents of the scheme would claim this is a 10% drop in rent - detractors that it's a 10% inflation of housing prices. This article makes no comment as to which of these it is.

Despite all this, owner-occupied investments are still viable for all but the lowest tax bracket agents who are still best served by sticking to shares. However, maximally leveraged owner-occupier returns are dismal at just 2%. Maybe that's not all too unsurprising in a high-interest world however.

Let's see what happens when we drop the interest rate a bit.

![Returns with CGT discount and 7.0% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-70.png)

_Returns with CGT discount and 7.0% interest_

At 7% interest we see the same major features as in the 9% case. At this point, it no longer makes financial sense for anyone in the bottom three tax brackets to own their own home. For those in the top two tax brackets, owner-occupied investments are still optimal financial choices, but only when they are bought with small loans - virtually loan-free for the 39% bracket, and around a third of the property value for the highest income agents. In other words, even if you're in the top tax bracket, it's not worth buying a home unless you can afford a ~65% deposit, at which point you should prioritize paying the remainder off before considering other investments.

This is where it also becomes apparent that the fully-leveraged owner-occupier investment still returns a dismal 2% yield - identical to the higher interest rate case. In fact, the maximally leveraged investment property returns (orange line) hasn't moved at all. This was one of the most surprising findings I made doing this modelling - that the equilibrium condition for almost all reasonable parameter estimates resulted in this phenomenon. I don't have a good intuitive explanation other than to say that's the way the maths pans out - that if you pin the final orange dot to the final green dot (equilibrium condition) then the gradient of the orange line is independent of the interest rate, hence the entire orange line does not move with interest rate changes. The blue line - the one related to housing investment _without_ a loan - changes as interest rates change, but the line related to housing investment with the _biggest possible loan_ does not. See the [equations in the appendix](#equations) if you need more convincing and aren't afraid of some symbolic manipulation.

In reality, interest rate changes will affect short term housing returns as the market adjusts. At equilibrium however, house prices and/or rents will have adjusted such that the after tax returns for housing investors will be the same as before the interest rate change.

![Returns with CGT discount and 6.2% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-62.png)

_Returns with CGT discount and 6.2% interest_

6.2% interest is the highest realistic interest rate that it makes financial sense for anyone to own their own home - and even then, only those in the highest tax bracket. Beyond this, it's financial optimal for the richest to own all properties and rent them out to everyone else. By that, I don't mean it's financially optimal just for the rich - it's financially optimal for _everyone_.

![Returns with CGT discount and 5.5% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-55.png)

_Returns with CGT discount and 5.5% interest_

At 5.5% interest rate, equilibrium rental yield is down to 1.1% - less than half of the 2.3% in the undiscounted case with the same interest rate. At this point though, with home ownership being such a poor option financially, it's highly questionable the extent to which this lower rental yield manifests itself in lower rents as opposed to higher property prices.

![Returns with CGT discount and 2.0% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-20.png)

_Returns with CGT discount and 2% interest_

We're pushing the bounds of believability with 2% interest rates, but I bring it up because this is the point at which our owner-occupier returns versus leverage (black line, right plot) is flat. Any interest rate lower than this and it will start going uphill. This is significant, because that's the first point at which allowing larger leverages can possibly result in better financial outcomes for participants. I will note however that the equilibrium rental yield at this point is -1.69%. This is net of maintenance costs, management fees and other premiums, taxes, and expenses, so it's at least conceptually possible that landlords aren't actually paying renters, but I'll accept we're pushing the limits of our assumptions here.

## Discount Distorted Phenomena

It's worth pointing out two distinct distortions the CGT discount introduces.

### High vs Low Interest Rates

There is an analogue of the high vs low interest rate regimes in the undistorted (no CGT discount) case. Recall that if the interest rate for greater than the pre-tax return on shares (8% with our figures), then housing market participants (both investors and owner occupiers) were better off paying down their mortgage. Interest rates below that level meant optimal behaviour was to stay as leveraged as possible. This makes sense: if the return you can get on some investment is less than the interest rate you're paying on a loan, you'd be wise to pay off the loan before putting anything into the alternative investment.

With the capital gains tax discount, we could say the analogue for owner occupiers is the point at which paying off the mortgage makes the overall housing investment better or worse. Graphically, this is the point where the blue and orange lines intersect at 0% nominal tax rate. As shown above, that's ridiculously low - around 2%. Historically, the only [record](https://www.rba.gov.au/statistics/tables/#:~:text=Housing%20Lending%20Rates%20%E2%80%93%20F6) I can find of rates this low were just after covid, and even then only on very restrictive products. Even these never dropped below 2%, nor did they stay at these levels for long.

Note that only in the low interest environment does giving home owners access to larger loans improve the returns of the investment. Given the infeasibility of such low interest rates under these model assumptions, this suggests increasing potential leverage will always result in worse financial outcomes for recipients.

### Ownership Optimality

In the undistorted case, an owner-occupier investment was always optimal - either maximally leveraged in the low interest case, or fully paid off when interest rates were high. Sure, most people don't have the funds to purchase a home outright, so under high interest rates people may not be able to realize that investment, but there was always a critical amount of leverage that made owning your own home optimal.

_This is not the case with a capital gains tax discount_. Graphically, this can be seen in cases where the blue dotted line is above the green dot in the appropriate tax bracket. At interest rates above 8.25%, full paid off home ownership is a good investment for all but those in the tax free bracket. Lower than 8.25% however, home ownership is no longer an optimal financial decision for those in the 18% tax bracket. At 7.3%, those paying 32% tax suffer the same fate, and by 6.3% even those in the highest tax break are better off selling up and purchasing an investment property and/or shares.

#### Optimality, not Affordability

It's important to note this whole article relates to optimizing financial decisions relating to housing, not on housing affordability. If Thanos snapped his fingers and half the population disappeared, the survivors - distraught as they would be - might find some small comfort in more affordable housing. This would affect both rents and house prices however, and once the market had adapted to the shock and returned to equilibrium, the dynamics and optimal investment policies described in this article would be no less true.

## Are we There Yet?

The RBA's official cash rate is currently 3.6%, with home loans sitting around 2% above that. Most economists expect that to either remain the same or drop by potentially 0.25% next year. This is also close to the [neutral rate estimates](https://www.rba.gov.au/publications/smp/2024/aug/financial-conditions.html). This puts us solidly in the higher interest rate regime, with all owner-occupied investments distinctly suboptimal under the sustainability/equilibrium assumptions of the model.

Now "under the sustainability/equilibrium assumptions" is a very large asterisk, and I'm not trying to say we're currently there. This model makes no attempt to deal with supply or demand shocks like the immigration-induced wave of demand Australia has faced since reopening borders after covid. In the short term, residential real estate investments may very well out-perform. This analysis says nothing about how we get to equilibrium - just the destination we're headed towards.

Having said that, the equilibrium net rental yield at 5.5% interest rates is a measly 1.1%. If we use the [50% rule](https://bassorealestate.com.au/buying-tips/whats-the-50-rule-in-real-estate/) as a very basic estimate, this corresponds to a gross 2.2% yield. This is a a good deal lower than the [current 3-5% average gross rental yield](https://australianpropertyupdate.com.au/apu/investor-loans-approach-record-levels-amid-tight-rental-market), indicating that rental yields will have room to fall to reach equilibrium. This analysis says nothing as to whether this manifests as increased property prices or decreased rents, but the current housing shortage and population growth would suggest one is significantly more likely than the other.

## Simulations

To validate the above claims about optimal behaviour, we can run some cash flow simulations. We consider 6 investment policies over a 30 year time span:

- _shares only_: never purchase property, i.e. rent forever;
- _interest only_: purchase a home at maximum leverage and only pay interest;
- _fixed term_: purchase a home with a standard fixed term mortgage;
- _repay ASAP_: purchase a home and pay it off as soon as possible;
- _delayed buy_: rent until shares investment allows a purchase with a loan at the critical leverage value discussed previously, then buy and repay the loan as fast as possible; and
- _delayed sell_: purchase a home and pay off the loan according to the fixed term schedule, but sell as soon as the leverage of the loan hits the critical leverage value discussed previously.

In all cases, we assume the same total cash flow, initially equal to the fixed term repayments increasing in line with wage growth. All excess income beyond rent/loan repayments is invested in shares, and in the delayed sell case, the proceeds of the home sale are immediately invested in shares. All values are normalized to the initially value of the house (i.e. a final net worth of 5 implies the agent finished the simulation with a net worth 5 times the house's original value).

For an agent in the highest tax bracket and a loan at 7% interest rate, the cash flows and net worth over time are given below.

![Cash flow simulations for highest tax bracket agent with 7% interest rate](/assets/img/posts/cgt-market-distortions/cashflows-base.png)

_Cash flow simulations for highest tax bracket agent with 7% interest rate_

If we look at the final net worths of each version, we see the results are consistent with the theory above.

![Final net worths for highest tax bracket agent with 7% interest rate](/assets/img/posts/cgt-market-distortions/net-worths-bar-base.png)

_Final net worths for highest tax bracket agent with 7% interest rate_

If that's not convincing, we can perform the simulation with each "delayed buy" and "delayed sell" for each possible time step and plot the final net worths.

![Final net worths for different purchase/sale times](/assets/img/posts/cgt-market-distortions/net-worths-plot-base.png)

_Final net worths for different purchase/sale times for highest tax bracket agent with 7% interest rate_

Again, this is entirely consistent with the above theory.

If we decrease the interest rate to 5.5% we see our investor is always better off investing in shares.

![Final net worths for highest tax bracket agent with 5.5% interest rate](/assets/img/posts/cgt-market-distortions/net-worths-plot-bad.png)

_Final net worths for highest tax bracket agent with 5.5% interest rate_

Again, this is entirely consistent with the above theory.

## Government Interventions

### 5% Deposit Scheme

As the name suggests, the [5% deposit scheme](https://firsthomebuyers.gov.au/australian-government-5-percent-deposit-scheme) gives first home owners the chance to buy their own home with a 5% deposit (or 2% for single parents) with the federal government taking the lender's mortgage insurance risk.

As discussed above, the only circumstance in which accessing a greater deposit for owner occupiers will result in a better financial outcome is if interest rates are unrealistically low for the duration of the loan - and even then, returns will only be better than shares while leverage is maintained. If a repayment schedule is enforced that eventually results in the home being paid off, optimal behaviour would be to purchase the home until some critical leverage value is reached, then sell out and return to renting.

### Help to Buy / Shared Equity Scheme

The [help to buy scheme](https://firsthomebuyers.gov.au/australian-government-help-buy-scheme) involves the government co-owning housing with first home buyers, stumping up 30% (or 40% for new builds) of purchase price and requiring only a 2% deposit ($\lambda=30$ or $\lambda=35$ in our model). It's not clear to me how ongoing costs like rates and insurance are handled, but for this model I'm assuming the scheme does not include any assistance with these.

There are very narrow circumstances in which these might help first home buyers into housing. Below are the returns when interest rates are at 6%. 6% was chosen because it's the lowest half-integer percentage value that makes _any_ use by eligible users optimal.

![Shared equity returns vs. leverage with interest rates at 6%](/assets/img/posts/cgt-market-distortions/shared-equity-returns.png)

_Shared equity returns vs. leverage with interest rates at 6%_

The negative slope of the brown line (returns with 30% government equity) indicates that this investment gets better as you pay off the loan. For highest tax bracket owner-occupiers, optimal behaviour is to buy in at a leverage of around 3.5. For those in the 39% tax bracket, buying your own home with this scheme is only close to financially optimal if the remaining 70% can be bought without a loan. Having said that, people in either of these tax brackets are ineligible for the scheme. All eligible participants are in a tax bracket that would make utilizing this scheme financially suboptimal regardless of the timing of property purchase/sale.

The 40% co-ownership line is _slightly_ better. It's sloping up, indicating that participants want to remain as highly leveraged as possible. For those in the lowest tax bracket, this is an optimal investment until the leverage reaches $\lambda=10$. Note that because the government owns 40%, this corresponds to an equity amount of 6% of the purchase price of the home. For those in the 18% tax bracket, this investment remains optimal until $\lambda=5$ or an equity stake of 12% of the value of the home. Below is a plot of cash flow simulations.

![Shared equity final net worths for an agent in the 18% tax bracket, 6% interest rate, 40% government equity](/assets/img/posts/cgt-market-distortions/net-worths-plot-shared-equity-40.png)

This shows a marginally improved financial outcome, but only if the recipient sells the property within roughly 8 years - otherwise they're better off utilizing shares throughout. Note this doesn't include any transaction costs, so the reality would be worse.

Those in the 32% tax bracket may benefit from holding this property even once it's paid off, though note only the lowest income earners in this bracket applying with a partner with negligible income are eligible for the scheme.

### First Home Owner Grant

The states have a myriad of first home owner grants available. For illustrative purposes, I'll consider the most generous form of [Queensland's](https://qro.qld.gov.au/property-concessions-grants/first-home-grant/eligibility/), where \\$30,000 is given to a newly built first home with value under \\$750,000. We'll assume a house value of \\$600,000 and a tax payer in the highest tax bracket with today's interest rate of 5.5% and a 20% deposit.

![Final net worths for a first home builder's grant in the highest tax bracket with 5.5% interest](/assets/img/posts/cgt-market-distortions/net-worths-plot-grant-05.png)

_Final net worths for a first home builder's grant in the highest tax bracket with 5.5% interest_

Clearly, the optimal behaviour is to build the home, collect the grant and resell it immediately (assuming you can sell it for what you paid for it plus the grant), then renting forever. Delaying sale always leaves you worse off, and if you wait more than 8 years you'd have been better off skipping it entirely. Similarly, if you buy a home on the last day of the simulated period, you'll net yourself a net worth higher by the grant value than if you hadn't (though going forward you'll be netting a lower return than if you hadn't, unless you immediately sell). In all other circumstances, you're better off just sticking to shares.

## Interventions That Might Work

Having established that the governments' current schemes don't actually change the behaviour of rational agents at equilibrium to promote long term home ownership, let's investigate some other policies that aren't actually being considered but might actually have an affect on long term outcomes.

### 25% CGT Discount

Reducing the capital gains tax discount to 25% has been considered by [government](https://www.pbo.gov.au/sites/default/files/2023-03/PER414%20%20ALP%20%20Negative%20gearing%20and%20capital%20gains%20tax%20CGT%20reform.PDF) and recommended by [the Grattan Institute](https://grattan.edu.au/wp-content/uploads/2023/04/Grattan-Back-in-Black-1.pdf). To be clear, there is no evidence that any major political party is currently considering a change to the CGT discount - but we'll still consider it here because it's the only thing that comes close to a feasible redress.

This lessens the distortions, and under the other model assumptions used here means housing investors end up with a very slightly positive effective tax rate. Under the current interest rate of 5.5%, this makes levered home ownership optimal for people in the highest tax bracket, and fully paid off home ownership optimal for those in the second highest tax bracket.

![Returns with 25% CGT discount and 5.5% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-55-25-cgt.png)

_Returns with 25% CGT discount and 5.5% interest_

This will obviously encourage home ownership beyond what's implied by the current system, but the slice of the population that's viable for (people earning over \\$135,000-\\$190,000 who can afford to buy a home outright, or those earning more than \\$190,000) is fairly slim.

### 50% Investment Income Discount

The argument for having a CGT discount is that capital appreciating assets are already "taxed" by inflation, and that they shouldn't be double taxed, i.e. only the increase in value beyond inflation should be explicitly taxed. Prior to the CGT discount an indexing method was used, but that was an accounting nightmare and I don't believe anyone would like to go back to that. Having said that, I fail to understand why this argument doesn't apply to other investment related gains and expenses like rent, dividends, or interest payments. If we simply applied it across the board, the end result would be a reduction in the tax rate by 50% for investment income. This would still make the system somewhat regressive, but it would completely eliminate the market distortions.

![Returns with 50% tax discount applied to all investment income/deductions and 5.5% interest](/assets/img/posts/cgt-market-distortions/equilibrium-interest-55-all-tax-discount.png)

_Returns with 50% tax discount applied to all investment income/deductions and 5.5% interest_

A few things to note here compared to the 50% CGT discount case with the same interest rate:

- each investment option gives a higher after-tax return for every tax bracket;
- agents at each tax bracket have the same optimal choice;
- the optimal choice is to leverage your own home first, and pour additional funds beyond a certain leverage point into shares;
  - the critical leverage point depends on your tax rate; and
  - even if an owner-occupied dwelling is paid off in full (suboptimal for all tax brackets), the returns are better than in the CGT discount case;
- more leverage for owner-occupier investments (i.e. lowering deposit requirements) would improve returns.

In every case, at equilibrium, every investor/owner occupier is better off. Now this might sound unsurprising given that it's the result of the government broadening a tax discount scheme. What is surprising is that this doesn't necessarily mean the government will collect less revenue. Every investor in this case is paying a positive effective tax rate. If we assume most investment comes from those in the top tax bracket - investors who could previously have been paying zero tax through careful weighting of shares and negatively geared property - there's every reason to believe the government would also collect more in tax revenue. The main losers in this case would be the banks, as investors are no longer inclined to borrow money to invest in assets with returns lower than the interest rate charged.

One thing this model doesn't address is what happens to rents. Note this change results in equilibrium net rental yield of 2.3%. Assuming non-interest operating costs of 1%, that means a gross rental yield of 3.3% - which is roughly consistent with [today's house market](https://www.firstlinks.com.au/asset-class-australia-offers-best-value-now). It's quite a bit higher than the net equilibrium yield with the normal CGT discount of 1.1%. This suggests that if we were to make the change shortly the effect on rental yields will be for them to stabilize, but enacting it later would result in a more dramatic change.

#### Do the Wealthy Need More Tax Breaks?

Now I'll stress I'm not advocating this as the tax policy. There's a strong argument that we should be moving towards a more progressive tax policy, rather than giving more discounts to those with enough excess income to invest. I'll preempt this criticism with two points.

Firstly, the discount doesn't need to be 50%. It could be 40%, or 25%. It could start at 50% and get reduced subsequently. The market distortions are eliminated with any value. It might be easiest to start at 50% because there's no way it can be misconstrued as a tax increase.

Secondly, and much more importantly, I don't think we should let perfect get in the way of good. Like it or not, the wealthy have undue influence on policy, and a solution that leaves them no worse off while benefitting the whole of society would be better than failing to enact a more just system. I see it akin to compensating slave owners when slavery was abolished. Was it _fair_ that slave owners got paid to release their slaves, when the slaves themselves got nothing? Absolutely not - but without the slave owners on board, slavery would have persistent longer than it did. It's certainly uncomfortable, but sometimes we have to take the least uncomfortable option.

## What if I'm Talking to a Normal Human?

Now all this is well and good, but when the discussion comes up in the pub, pulling out graphs and discussing gradients and intercepts isn't going to win over many people. Trust me, I've tried. In this section I'll try and distill what the above says into an elevator pitch without pictures or the accompanying 5,000 word essay.

### The CGT Discount is a Tax Discount in Name Only

On an investment basis, the CGT discount and negative gearing result in housing investors having a negative effective tax rate. We don't normally talk about negative tax rates - we call them subsidies. The higher the investor's nominal tax rate, the larger the subsidy. This means the highest tax rate investors can out-compete all lower tax rate participants when using leverage. Assuming there is another asset class where this advantage doesn't exist, investors not in the highest tax bracket will always be better off (in expectation) opting for the other investment.

### Lower Interest Rates Don't Increase Affordability

For those yet to enter the housing market, lower interest rates drive more capital into the market from investors. Regardless of whether this drives up prices on existing properties or lowers rent by increasing supply, the end result is a lower return for prospective owner-occupiers.

### More Leverage Won't Fix Things

Increasing leverage for investors exacerbates the problem - that should come as no surprise. However, increasing leverage for owner occupiers _also_ makes the financial decision worse except under extremely low interest rates that have only been seen in Australia once (immediately after covid hit) and were never intended to last long.

### This isn't Irrational Exuberance

It's tempting to look at a market where renting is always preferable to buying and dismiss it as the result of irrational exuberance, FOMO, property speculators and/or ponzinomics that will eventually collapse. The reality is worse. Rational behaviour of all agents purely motivated for financial gain with Australia's tax settings would result in 0% owner-occupied home ownership under neutral interest rates.

### Home Ownership Is a Luxury

There are certainly non-financial benefits to owning your own home. I haven't discussed these non-financial benefits throughout this paper, but that does not mean they don't exist. In reality, many people will buy their home despite it being suboptimal financially. I'm not saying that's a bad idea - just that it's potentially expensive. Luxury cars are expensive, but that doesn't stop people buying them. Maybe think twice before taking out a loan with a 5% deposit to purchase one though.

### Housing Swaps: Ownership with [Extra Steps](https://www.youtube.com/shorts/vHI8XIE_vWE)

If you're in the top tax bracket, and you have a friend in the top tax bracket, at some point you'll be better off buying their property and renting it to them, and letting them buy your property and renting it off them. I'm not saying this is a good outcome - but it's the outcome we deserve for electing politicians who refuse to fix the system.

## Conclusions

That was really long. Thanks for sticking with me - it was a much deeper rabbit hole than I expected it to be. If that's still not enough, I've included some more [figures](#figures-that-didnt-fit) below that I generated but didn't quite fit the narrative. Enjoy.

## Appendix

### Symbols

|Symbol|Default Value|Description|Source|
|-----|-----|---|---|
| $r$       | $5.5%$     | Interest rate for loans   | [CBA](https://www.commbank.com.au/home-loans.html)
| $w$       | $3.7\%$    | Wage growth inflation                                 | [Treasury estimates quoted by ABC](https://www.abc.net.au/news/2025-11-02/house-prices-climbing-for-generation/105954416)
| $c_h$     | $w$       |Housing capital appreciation                          | Steady state implications|
| $y_h$     |           |Housing rental yield                                  |            |
| $y_h^*$   |           |Equilibrium housing rental yield                      | Market equilibrium assumption |
| $c_s$     | $5\%$     |Shares capital appreciation                           | [See below](#stock-return-estimates) |
| $y_s$     | $3\%$     |Shares dividend yield                                 | [See below](#stock-return-estimates) |
| $\lambda$ |           |Loan leverage, e.g. 20% deposit implies $\lambda = 5$ |            |
| $\lambda_m$| $5$      | Maximum leverage available to investors              |            |
| $i$       | $5.5\%$   | Interest rate                                        | [CommBank](https://www.commbank.com.au/home-loans.html)                |
| $\Delta$  | $0.5$     | Capital gains tax discount                           | [ATO](https://www.ato.gov.au/individuals-and-families/investments-and-assets/capital-gains-tax/cgt-discount) |
| $t$     | [See table](#a-progressive-base-tax-system) | Nominal marginal tax rate                                        |  |
| $t_m$     | $47\%$ | Highest nominal tax rate                                        |  |
| $R_o$     | | Total home owner return | |
| $R_h$     | | Total non-owner-occupier housing investor after-tax return | |
| $R_s$     | | Total share investor after-tax return |  |

#### Stock Return Estimates

The stock return estimates $c_s = 5\%$ and $y_s = 3\%$ were based on the upper end of [Vanguard estimates](https://corporate.vanguard.com/content/corporatesite/us/en/corp/vemo/vemo-return-forecasts.html) for the next 30 years (all equities estimates in the 3-8% total) but below historical returns (e.g. 13% for [VGS](https://www.vanguard.com.au/personal/invest-with-us/etf?portId=8212&tab=performance) since 2014). The capital / distribution split was roughly based on [VGS](https://www.vanguard.com.au/personal/invest-with-us/etf?portId=8212&tab=prices-and-distributions) which has historically distributed ~2.5% annually.

### Equations

General after tax levered returns

$$
R(\lambda, t; c, y, r) = \lambda [(1 - (1 - \Delta)t)c + (1 - t)y] - (1 - \lambda)(1 - t)r
$$

Investment housing returns:

$$
R_h(\lambda, t; r) = R(\lambda, t; c_h, y_h, r)
$$

Shares returns:

$$
R_s(t) = R(1, t; c_s, y_s, 0)
$$

Owner occupier housing returns:

$$
R_o(\lambda; r) = R_h(\lambda, 0; r)
$$

Steady state implications:

$$
c_h = w
$$

Market equilibrium:

$$
y_h^* = y_h \text{ such that } R_h(\lambda_m, t_m) = R_s(t_m)
$$

which after substitution yields

$$
y^{*}_{h} = \frac{- \Delta c_{s} t_{m} + \Delta \lambda_{m} t_{m} w + c_{s} t_{m} - c_{s} - \lambda_{m} t_{m} w + \lambda_{m} w + r \left(\lambda_{m} t_{m} - \lambda_{m} - t_{m} + 1\right) + t_{m} y_{s} - y_{s}}{\lambda_{m} \left(t_{m} - 1\right)}.
$$

Substituting $y_h = y_h^*$ and $\lambda=\lambda_m$ into our levered investment property returns yields

$$
R_{hm}^*(t) = R(\lambda_m, t; w, y_h^*, r) = \frac{- \Delta c_{s} t_{m} + \Delta \lambda_{m} t_{m} w + c_{s} t_{m} - c_{s} + t \left(\Delta c_{s} t_{m} - \Delta \lambda_{m} w - c_{s} t_{m} + c_{s} - t_{m} y_{s} + y_{s}\right) + t_{m} y_{s} - y_{s}}{t_{m} - 1}.
$$

Note this is independent of interest rate $r$, meaning the orange lines in the returns plots are the same for all interest rates.

### Why Not Higher?

In the [CGT Discount Distortions](#cgt-discount-distortions) section we started at an interest rate of 9%. If we increase it beyond this, very soon our unlevered housing investment line (blue) intersects our shares line (green). Our efficient market assumption would suggest that would mean the impact of lower tax bracket households should start out-competing high income households for fully paid off housing. Now maybe there's a world where interest rates remain at or above 10% for an extended period and house prices are cheap enough that those on less than \\$45,000 income can afford to purchase investment homes without loans in sufficient quantities that they move the market - but it's not this world.

### Figures That Didn't Fit

#### Tax Neutral Weightings

For investors in the top tax bracket, property ownership has a negative effective tax rate (orange lines have positive gradient). To realize this negative effective tax rate, the agent needs income from another source. In the below plot, we consider a portfolio made up of $\alpha$ shares and $(1 - \alpha)$ in maximally leveraged real estate (resulting in $\lambda(1-\alpha)$ total pouring into property).

![Tax neutral portfolio weighting](/assets/img/posts/cgt-market-distortions/tax-neutral-weighting.png)

_Tax neutral portfolio weighting_

It shows that with default parameters, having 57% or more invested in shares results in an average tax rate of 0%. A higher weighting would give the same after tax return, though less would require more income to offset, e.g. from a salary.

#### Required Leverage For All Interest Rates

![Required leverage values for equal home ownership and shares returns](/assets/img/posts/cgt-market-distortions/even-leverage-vs-interest-rate.png)

_Required leverage values for equal home ownership and shares returns_

The above shows the conditions for which an agent would prefer to purchase their own home compared to investing in shares. Any point below and right of the relevant tax bracket line would be better off in home ownership. Fo Example, at 8% interest, agents in the 32% tax bracket would only buy their home if they had the capital to use $\lambda < 1.5$, while those in the 39% tax bracket would buy in around $\lambda = 1.8$. The lower tax brackets would never purchase their own home at this interest rate if acting financially optimally.

#### When Optimizing Returns Means Minimizing Tax

Here we consider the after tax returns of an agent offered two investments: a base investment returning 5% capital gains and 3% regular income, and a potentially levered investment returning 3% capital and 3% regular income with interest charged at 6.5%. The agent can take any amount of each asset, but their average tax rate over the weighted investment must be non-negative.

![Returns for a weighting of two investments constrained to having a non-negative average tax rate](/assets/img/posts/cgt-market-distortions/simple-weighted-returns.png)

_Returns for a weighting of two investments constrained to having a non-negative average tax rate_

In this case, returns are optimized on the zero effective tax frontier. This feels like a quadratic programming problem, and there are probably conditions under which return maximization coincides with tax minimization which could be interesting to explore... but that's a project for another time.

## Introduction

Triangular sails were invented in the 2nd century, and since then, determining optimal routes for boats has been a key question not just for racing, but also for commerce, avoiding pirates, and catching commercial vessels that are avoiding you.

Now that cargo vessels are beginning to explore adding sails to make their travel more efficient, optimal sailing routes for commerce are a key issue in global supply chain.

In this project we work to solve the related problem by modeling the optimal route for a sailboat race in lake Michigan. All of the data that we use is from a real race day in July 21st 2024.

We started from the simplest case and then increased the model complexity from there. We have a few more parameters to add, including land penalties to prevent the ships from going on land, which we hope to implement soon. We also want to constrain the rate at which the angle of the boat changes, and we want to have wind and current impact boat acceleration instead of velocity.

## Approaches

We started with a simple model that is impacted by current but moves with a constant velocity; i.e. a motorboat. This problem was easy to solve analytically.

The next model examined sailing dynamics, looking at a simple model where wind directly impacts the velocity of the boat, pushing it in the same angle as the sail. This method was not able to rediscover real-world sailing techniques, since it didn't enable sailing upwind.

The final model that we have examined involved looking at real-world, physically motivated sailing. Sails act somewhat like airplane wings, pulling boats as wind moves faster over their front than their back, enabling ships to go upwind using techniques called tacking and jibing.

# *Magic* and Neural Nets: Classifying Trading Card Artwork with Tensorflow and CNNs
For my third Capstone project, I wanted to further explore computer vision and image processing using Convolutional Neural Networks with Keras and Tensorflow. As inspiration for the project, I decided to revisit the subject of my [first Capstone project](https://github.com/n-sweep/galvanize_capstone_1), the trading card game *Magic: the Gathering*. I wanted to find whether a Convolutional Neural Network could be trained to recognize the color of a card based only on the card's artwork (more on that below.) Already familiar with the API at [Scryfall.com](https://scryfall.com/docs/api), I knew that not only did they provide access to quality scans of the trading cards I wanted to classify, but their available data also included image files of the artwork alone, with the rest of the card cropped away - exactly what this project required. 

### Magic: the Gathering
Concieved and developed by mathematician Richard Garfield, *Magic: the Gathering* (commonly known as *MTG* or simply *Magic*) is a trading card game released by Washington State-based game publisher *Wizards of the Coast* in 1993. Each card is printed with text and various symbols explaning how the card is used, as well as a unique piece of artwork to help bring the card to life. 

In the mid 90s, I was too young to really grasp the mechanics of the game but I had a family member, about 10 years my senior, who collected and played. I eventually rediscovered the game and played for many years, but I have fond memories of looking through his cards and admiring their fantastic artwork.

# Data

![distribution of colors](img/plots/color_dist.png)
<img src="img/colors_inline_transparent.png" alt="mtg color symbols" width="575"/>

Over *Magic*'s nearly 30-year history, *Wizards of the Coast* has printed over 25,000 unique cards. They always keep balance in mind when designing new sets of cards, which leaves us with a conveniently pre-balanced distribution of colors across our data set.

*Magic* is a game of color: White, Blue, Black, Red and Green are core to how the game works and how it is designed. Color is often a player's first consideration when building a deck, a card's color tells what resources you need to play it and, on the subject of this project, the color can influence the art created for the card.

So, **can we train a neural network to predict the color of a card based on the artwork created for that card?**

Our neural net will consider ~19,000 images of *MTG* artwork gathered from Scryfall.com's convenient [API](https://scryfall.com/docs/api) and try to predict which of *Magic*'s five colors that card belongs to.

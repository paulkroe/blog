---
title: LinkedPool
date: 2024-09-01
author: Paul Kroeger
description: Web Application around the LinkedIn API
toc: true
tags:
---
## The Idea
Upon starting at Columbia University, introductory lectures often ended with instructors sharing their LinkedIn profiles, inviting students to connect. While connecting with lecturers was great, I found myself more interested in networking with my peers. To address this, I thought of a web application where users could scan a QR code that links to a site displaying the LinkedIn names and profile pictures of others who had scanned the same code. The concept included a feature to connect directly via a simple button next to each name. 
From the start, I knew the application needed to be as seamless as possible to have any chance of being used. With this intuition in mind, I coined the name LinkedPool, sought a logo from DALL-E, and began the development process.

<div style="text-align: center;">
<img src="/LinkedPoolLogo.png" alt="LinkedPool Logo" width="100" height="100">

*I used a logo only slightly modified from what DALL-E gave me*
</div>


## Implementation
Before working on LinkedPool, my JavaScript experience was minimal—I had written fewer than 50 lines of JS code and had never used Express. My knowledge of HTML and CSS was similarly insufficient. Fortunately, prompted correctly, ChatGPT 4 is capable of generating enough boilerplate code to piece together a satisfactory prototype.
For me personally, the greatest challenge was interfacing with the LinkedIn API. An update to the LinkedIn API in 2022 meant that OpenAI models referred to outdated endpoints. Despite these hurdles, I am still greatly impressed with the capabilities of the models.

After identifying the correct endpoints, with the assistance of ChatGPT, I rapidly implemented a working lobby system. This enabled the retrieval of users’ names and LinkedIn profile pictures.

<div style="text-align: center;">
 <img src="/lobby.png" alt="Creating a Lobby"> <i>The Lobby page I created for LinkedPool</i>


<img src="/LinkedInLogin.png" alt="LinkedIn Login">
<i>Eventually, I managed to implement LinkedIn's three-legged OAuth flow.<br></i>

<img src="/FullLobby.png" alt="Populated Lobby">
<i>A fully populated lobby should look like this.<br></i>

</div>

Initially, I hoped to allow users to connect directly through a “Connect” button on the LinkedPool site. However, LinkedIn’s API restrictions prevented this. I considered adding a button linking directly to individual LinkedIn profiles as a workaround, but this functionality was also restricted under the basic API permissions. Faced with no viable alternatives, I resorted to having users manually copy and paste their LinkedIn URLs—a cumbersome process that likely undermines the usability of the project. While testing remains, I am concerned that this additional step kills the idea.

<div style="text-align: center;">
<img src="/pasteLink.png" alt="Pasting the LinkedIn URL">
<i>As a last resort, users had to manually paste their LinkedIn URL.<br></i>
</div>

## Final Thoughts
Even though I consider the project a failure at least for now, I am still happy with how it turned out. I learned a lot.
Besides a basic introduction to full-stack development, I realized two things.

- It was very interesting developing a whole project with ChatGPT 4. It allowed me to pay closer attention to its long-term memory and reasoning capabilities. Additionally, I noticed again and again how much time ChatGPT saved me. Having a personal tutor who can answer any of my very basic questions was immensely helpful and sped up my learning process significantly.

- Additionally, I learned firsthand how it feels if your whole idea builds on somebody else's infrastructure. Building around immutable constraints is a very different experience than building something from scratch. While I enjoyed the challenge, in the future I will be careful to not rely on third parties for the core functionality of my projects.
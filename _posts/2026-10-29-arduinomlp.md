---
layout: post
title: "Training a multi-layer perceptron on an Arduino"
date: 2026-10-29
category: "Embedded, Arduino"
description: Building and training a multi-layer perceptron on an Arduino UNO and displaying the neurons with LEDs
---

Visual odometry is the process of estimating a camera's movement by
analysing the visual information captured across consecutive frames.

I started exploring this while experimenting with a phone camera and
OpenCV. The basic idea sounds straightforward: detect visual features,
match them between frames, and use those correspondences to estimate
how the camera moved.

In practice, things get more complicated.

## The problem with rotation

When a camera rotates, features can move substantially across the
image even if the camera has barely translated through space.

That means image motion isn't necessarily the same thing as physical
displacement.

An algorithm that interprets every change in feature position as
evidence of forward movement can produce unreliable estimates.

## Why feature tracking isn't enough

Feature detection methods such as ORB can identify distinctive points
in an image. Matching those points across frames gives us information
about how the image changed.

But those matches alone don't automatically tell us how the camera
moved. The geometry of the camera, the scene, and the type of motion
all matter.

## What I'm exploring next

Some approaches worth investigating include:

- Selecting keyframes instead of processing every frame identically.
- Estimating rotational motion separately from translation.
- Improving feature matching and rejecting outliers.
- Understanding how camera calibration affects motion estimation.

The goal isn't just to make the visualisation look convincing. It's
to understand when the estimate is trustworthy and when it isn't.

That's the interesting part of building things: the gap between
something that appears to work and something you actually understand.
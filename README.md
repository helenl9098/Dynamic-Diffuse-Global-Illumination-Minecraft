Dynamic Diffuse Global Illumination in Mini-Minecraft (DDGI Minecraft)
======================

![](img/probe_vicinity_debug.png)

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* Janine Liu
  * [LinkedIn](https://www.linkedin.com/in/liujanine/), [Github](https://github.com/j9liu)
* Spencer Webster-Bass
  * [LinkedIn](https://www.linkedin.com/in/spencer-webster-bass/), [Github](https://github.com/spencerwb)
* Helen Liu
  * [personal website](http://liuhanyu.net/)


## Table of Contents
 * Introduction
 * Credits
 * Walkthrough
 * Results
 * Performance Analysis

## Introduction

Real-time raytracing has spurred the development of faster rendering techniques to achieve high-quality graphics without dramatically affecting game performance. One topic of study includes real-time global illumination: a recent paper by Majercik, et. al (2019) details a method of approximating indirect lighting, with information updated at every frame based on dynamic lighting and moving scene geometry. The paper, [*Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields*](http://jcgt.org/published/0008/02/01/paper-lowres.pdf), is the focus of our project; we aim to implement the algorihm described, while generating our own Minecraft-inspired scenes to test its efficacy in real-time. (We are inspired by a shared interest in Minecraft.)

## Credits

Due to hardware limitations, our team could not take advantage of the Vulkan raytracing extension. Thus, our code is based on the Realtime Vulkan Path Tracer (RVPT) developed [here](https://github.com/GraphicsProgramming/RVPT/), which relies on compute shaders to raytrace scenes. Without this, our Vulkan framework would have taken much longer to setup from scratch.

## Walkthrough

Here we lay out our interpretation of the paper and our implementation details.

### Light Field Probes

To approximate global illumination at points in a scene, we can utilize *light field probes* as described in [*Real-Time Global Illumination using Precomputed Light Field Probes*
](http://casual-effects.com/research/McGuire2017LightField/McGuire2017LightField.pdf) by McGuire et. al (2017). These probes conceptually are spheres dispersed throughout a scene in a grid-based *irradiance field*. This field can be made up of an arbitrary number of probes, but ideally it encapsulates all objects in the scene.

The implementation of these probes in Majercik et. al's paper differs from the Majercik et. al implementation. Before the scene is raytraced, each of the *m* probes sends out *n* rays that potentially intersect with the scene. We find the irradiance value at those intersections, i.e. the direct and indirect lighting at that point, and we also record the distance from the originating probe to that intersection. This information is stored in a texture; each pixel corresponds to a ray from a probe, and the pixels corresponding to one probe are packed together.

| ![](img/albedo_texture.png)  | ![](img/cave.png) |
| ------------- | ------------- | 
| An example of the probe ray texture. | A capture of the scene, for reference. |

### Sampling Probes

After the probes collect their data, the scene is raytraced in a compute shader and rendered to an image on the screen. For every ray that finds an intersection in the scene, we first find the direct lighting at that point. Then, we identify the eight closest probes to that point in the scene. The grid-based structure of the irradiance field allows every point in the scene to be encapsulated in some *probe cage*—a set of eight probes that form a cube surrounding the point.

| ![](img/probe_cage.png) | ![](img/probe_vicinity_debug.png) |
| ------------- | ------------- |
| A figure depicting the eight probes around a pixel on the green triangle, taken from the paper. | A debug view of the probe indices that correspond to pixels in the scene. The color corresponds to the index of the bottom, front, leftmost probe in the cage surrounding each pixel. |

Once the probe cage has been determined, we iterate over every probe in the cage and sample the light that the probe sees for that point. More specifically, we get the normal at the point of intersection, then get the ray of the probe that aligns with the direction's normal. In this way, we treat the probe as see-through; we get the irradiance that the probe sees in that direction, then add that to the total irradiance at that point. 

### Weights ###

### Scene Generation

## Results

*To be updated when the project is finished.*

## Performance Analysis
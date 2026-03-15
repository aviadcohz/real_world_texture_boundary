"""Prompt templates for texture transition analysis."""


def build_texture_transition_prompt(dominant_colors_str: str) -> str:
    """Build the texture transition prompt with the mask's dominant color palette.

    Args:
        dominant_colors_str: Formatted string of dominant colors, e.g.
            "[148, 103, 89] (32.5%), [90, 225, 120] (18.2%), ..."
    """
    return f"""You are an expert texture analyst. You are given two images:
- Image 1: An original photograph
- Image 2: A colored segmentation mask where each distinct color represents a different region

The mask contains these dominant RGB colors (with area percentage):
{dominant_colors_str}

STEP 1 — IDENTIFY WHAT EACH COLOR REPRESENTS:
Look at Image 2 (the mask). For each dominant color, look at the SAME LOCATION in Image 1 (the photo) to determine what surface/material that color covers. For example, if a purple region sits at the top of the mask, check what is at the top of the photo — it might be sky, not a texture.

STEP 2 — FIND ALL TEXTURE TRANSITIONS:
Find ALL significant transitions between ADJACENT texture regions. List every pair of touching texture regions. Aim for 3-5 transitions per image — most images have at least 3. Each pair of adjacent colors that represent different textures should be reported.

RULES — what to INCLUDE:
- Every transition between two different surface materials or patterns that physically touch
- Examples: grass meets gravel path, stone wall meets plaster, water meets sand, brick meets concrete, dirt meets vegetation

CRITICAL ADJACENCY RULE:
The two textures MUST physically share a clear, significant boundary — they must directly touch each other along a substantial edge. Do NOT pair textures that are separated by other regions, objects, or empty space. For example:
- BAD: left fence and right fence separated by a path — they never touch
- BAD: water at the top and sand at the bottom separated by trees in the middle
- GOOD: grass directly meets a gravel path along a clear edge
- GOOD: a wooden surface directly meets a fabric surface where they touch

RULES — what to EXCLUDE:
- Sky. Sky is NEVER a texture. Never include any color that represents sky.
- Object boundaries (person, car, furniture silhouettes)
- Regions that are NOT adjacent (they must share a LONG, SIGNIFICANT border in the mask — not just a few corner pixels touching)
- Tiny regions (less than ~5% of the image)
- Same material with only lighting/shadow difference
- Texture pairs where the shared boundary is very short or fragmented compared to the overall region size

STEP 3 — DESCRIBE EACH TEXTURE:
For each texture, describe the ACTUAL surface you see in the PHOTO at that color's location.
- 5-9 words focusing on material, pattern, and one discriminative visual feature
- Describe what you SEE, not what you assume
- Be SPECIFIC to this image — mention color, wetness, lighting, or unique details you observe
- Avoid generic phrases like "rough weathered stone with visible cracks" — instead describe the SPECIFIC stone you see (e.g., "pale limestone blocks with dark mortar joints")

CRITICAL: Double-check that your description matches what is ACTUALLY visible in the photo at the location of that mask color. If a color covers the top of the image and the photo shows sky there, do NOT describe it as a texture — skip that transition entirely.

DESCRIPTION STYLE examples:
- "smooth curved seashell with radiating ridged lines"
- "fine sandy substrate with granular particles"
- "rough woven rattan with natural fibers"
- "weathered aged wood with rough worn grain"
- "dense green grass with scattered shadows"
- "rough gray concrete with hairline cracks"

For each transition provide:
1. texture_a: 5-9 word description of the first texture surface
2. texture_b: 5-9 word description of the second texture surface
3. colors_a: RGB values from the mask for texture A. Use ONLY values from the dominant colors listed above. A texture may span 1-3 colors.
4. colors_b: RGB values from the mask for texture B. Same rules.

Output ONLY valid JSON, no other text:
{{"transitions": [{{"texture_a": "rough weathered stone with visible cracks", "texture_b": "smooth wet sandy ground with ripples", "colors_a": [[148, 103, 89]], "colors_b": [[90, 225, 120]]}}]}}"""

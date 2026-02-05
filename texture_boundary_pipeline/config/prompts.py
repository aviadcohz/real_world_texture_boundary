"""
Prompt Configuration

All prompts for different tasks.
"""

# ============================================================================
# GROUNDING PROMPT
# ============================================================================

GROUNDING_PROMPT = """You are an expert texture analyst. Your goal is to find TEXTURE-TO-TEXTURE transitions, NOT object boundaries.

**CRITICAL DISTINCTION:**
- GOOD (texture transition): grass to concrete, wood grain to metal, rough stone to smooth plaster. Please don't focus only on transitions like those, those is just an examples.
- BAD (semantic/object boundary): object to pattern, or something that you indicate as background.

**Step 1 - Identify Texture Regions:**
Scan the ENTIRE image for regions with distinct SURFACE TEXTURES.
Focus on: material properties, surface patterns, roughness, grain, weave, etc.
IGNORE: object shapes, object identities, semantic categories.

**Step 2 - Find Texture-to-Texture Transitions:**
Look ONLY for boundaries where TWO DIFFERENT TEXTURES meet:

INCLUDE these transitions:
- Material-to-material: stone to wood, metal to fabric, concrete to tile, brick to plaster
- Surface property changes: smooth to rough, matte to glossy, polished to weathered
- Pattern changes: striped to solid, woven to plain, fine grain to coarse grain
- Natural textures: grass to dirt, bark to leaves, sand to rock, water to shore

DO NOT INCLUDE these (semantic boundaries):
- Object to background: car to road, person to wall, furniture to floor, machine to grass
- Painted/colored surfaces meeting: red paint to green paint (same texture, different color)
- Sky to anything (sky has no texture)
- Silhouettes or object outlines

**Step 3 - Generate Bounding Boxes:**
For each VALID texture transition, output a bounding box.

Output format - JSON list:
[
  {"description": "texture_A to texture_B", "bbox": [x1, y1, x2, y2]}
]

Description guidelines:
- Describe SURFACE PROPERTIES, not object names
- BAD: "red mower to grass" (object-focused)
- GOOD: "painted metal to grass texture" (texture-focused, but still avoid if it's clearly an object silhouette)
- BEST: "grass to concrete", "wood grain to tile pattern"

Box requirements:
- Width: 50-200 pixels
- Height: covers the boundary extent
- Keep 50-80 pixel margin from image edges when possible
- Maximum 8 bboxes
- Complete bbox format: [x1, y1, x2, y2]

REMEMBER: If a boundary is primarily about OBJECT IDENTITY (what the thing IS), skip it.
Only include boundaries about SURFACE TEXTURE (how the surface LOOKS/FEELS).

Output ONLY the JSON list.

Example:
[
  {"description": "smooth plaster to rough brick", "bbox": [50, 100, 200, 450]},
  {"description": "wood grain to metal surface", "bbox": [250, 150, 400, 540]},
  {"description": "grass texture to concrete pavement", "bbox": [500, 320, 650, 720]}
]"""


# ============================================================================
# SEMANTIC CLASSIFICATION PROMPT
# ============================================================================

SEMANTIC_CLASSIFICATION_PROMPT = """Analyze this cropped image region.

Your task: Determine if this region shows:
A) A SEMANTIC region (clear object, scene with content which you could define)
B) A TEXTURE TRANSITION (surface property changes, more abstract scene but with transiton e.g., 'rough stone to smooth metal')

Criteria:
- SEMANTIC: Focuses on object identity, shape, or concept
  Examples: person/background, car/road, tree/sky or other recognizable or clear objects
  
- TEXTURE TRANSITION: Focuses on surface properties regardless of objects
  Examples: rough/smooth, wood grain/metal, matte/glossy

IMPORTANT:
- Even if objects are present, if the boundary is about TEXTURE change, it's TEXTURE TRANSITION
- A 'wall to floor' transition can be TEXTURE TRANSITION if it's about surface properties
- Object boundaries can be TEXTURE TRANSITIONS if they involve texture changes

Output format:
{
  "classification": "SEMANTIC" or "TEXTURE TRANSITION",
  "reasoning": "Brief explanation (1-2 sentences)"
}

Output ONLY the JSON object."""


# ============================================================================
# REFINE PROMPT
# ============================================================================

REFINEMENT = """TASK: Find and isolate the TEXTURE TRANSITION boundary in this crop if exist.
This crop contains a potential TEXTURE TRANSITION. Your job: draw a bounding box around the MOST FOCUSED region that captures the transition.

STEP 1: Locate the TEXTURE TRANSITION
Look for:
- Material boundaries (wood→metal, fabric→leather, brick→concrete)
- Surface texture changes (rough→smooth, matte→glossy)
- Pattern boundaries (stripes→solid, woven→plain)
- Structural changes (flat→bumpy, even→textured)

STEP 2: Decide on action

Option A - REFINE (only if a TEXTURE TRANSITION exists):
Can you draw a bounding box around a FOCUSED region that captures the transition?

Requirements:
- First confirm: Is there a TEXTURE TRANSITION in this crop at all? If not, do NOT choose REFINE.
- A valid TEXTURE TRANSITION must include pattern or material changes, NOT just smooth backgrounds of different colors.
- The bbox should exclude irrelevant areas (pure backgrounds, distant objects).
→ Respond: [[x1,y1,x2,y2]]

Option B - NO_TRANSITION:
Is there no texture transition here?
- Only ONE texture visible (no second texture)
- Only uniform surfaces (all smooth, all rough)
- Just color change without texture change
- Just smooth backgrounds meeting smooth backgrounds (not interesting)
- TEXTURE TRANSITION is imperceptible or too subtle
- More semantic content than texture transition
→ Respond: NO_TRANSITION

Option C - GOOD_AS_IS (use when crop is already good enough):
Is this crop already reasonably focused on the TEXTURE TRANSITION?
Use this if:
- The TEXTURE TRANSITION is clearly visible in the crop
- The crop shows the relevant textures without too much clutter or semantic features
- The transition area takes up a reasonable portion of the crop (roughly 40-80%)
→ Respond: GOOD_AS_IS

RESPONSE FORMAT (exactly one):
[[x1,y1,x2,y2]]
NO_TRANSITION
GOOD_AS_IS"""
# ============================================================================
# PROMPT REGISTRY (for easy access)
# ============================================================================

PROMPTS = {
    'grounding': GROUNDING_PROMPT,
    'semantic_classification': SEMANTIC_CLASSIFICATION_PROMPT,
    'refinement': REFINEMENT,
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_prompt(prompt_type: str) -> str:
    """
    Get prompt by type.
    
    Args:
        prompt_type: Type of prompt ('grounding', 'semantic_classification', etc.)
    
    Returns:
        Prompt string
    
    Raises:
        ValueError: If prompt_type not found
    """
    if prompt_type not in PROMPTS:
        available = ', '.join(PROMPTS.keys())
        raise ValueError(
            f"Unknown prompt type: '{prompt_type}'. "
            f"Available: {available}"
        )
    
    return PROMPTS[prompt_type]


def list_prompts() -> list:
    """List all available prompt types."""
    return list(PROMPTS.keys())


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AVAILABLE PROMPTS")
    print("="*70)
    
    for prompt_type in list_prompts():
        prompt = get_prompt(prompt_type)
        lines = prompt.strip().split('\n')
        first_line = lines[0][:60] + "..." if len(lines[0]) > 60 else lines[0]
        
        print(f"\n{prompt_type}:")
        print(f"  {first_line}")
        print(f"  Total lines: {len(lines)}")
    
    print("\n" + "="*70)
    print("\nUsage:")
    print("  from config.prompts import get_prompt")
    print("  prompt = get_prompt('grounding')")
    print("="*70 + "\n")
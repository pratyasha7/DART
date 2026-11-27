use hdk::prelude::*;
use acoustic_app_integrity::*;

/// Creates an entry on the DHT AND links it to a static anchor ("all_claims")
#[hdk_extern]
pub fn create_triangulation_claim(claim: AcousticTriangulationClaim) -> ExternResult<ActionHash> {
    // 1. Commit the entry to local source chain and DHT
    let action_hash = create_entry(EntryTypes::AcousticTriangulationClaim(claim))?;

    // 2. Create typed path anchor for "all_claims"
    let path = Path::from("all_claims");
    let typed_path = path.typed(LinkTypes::AllClaimsAnchor)?;
    
    // 3. Ensure path anchor exists in DHT and get its hash
    typed_path.ensure()?;
    let anchor_hash = typed_path.path_entry_hash()?;

    // 4. Link the anchor to the newly created entry's action hash
    create_link(
        anchor_hash,
        action_hash.clone(),
        LinkTypes::AllClaimsAnchor,
        (),
    )?;

    Ok(action_hash)
}

/// Fetches all gunshot claims linked to the "all_claims" anchor across the DHT
#[hdk_extern]
pub fn get_all_claims(_: ()) -> ExternResult<Vec<AcousticTriangulationClaim>> {
    let path = Path::from("all_claims");
    let typed_path = path.typed(LinkTypes::AllClaimsAnchor)?;
    typed_path.ensure()?;
    let anchor_hash = typed_path.path_entry_hash()?;

    // Fetch all links pointing away from this anchor
    let links = get_links(
        GetLinksInputBuilder::try_new(anchor_hash, LinkTypes::AllClaimsAnchor)?.build(),
    )?;

    let mut claims = Vec::new();

    for link in links {
        // Convert AnyLinkableHash target into ActionHash safely
        if let Some(action_hash) = link.target.into_action_hash() {
            // Fetch Record from DHT
            if let Some(record) = get(action_hash, GetOptions::default())? {
                // Try converting the Record reference directly into AcousticTriangulationClaim
                if let Ok(claim) = AcousticTriangulationClaim::try_from(&record) {
                    claims.push(claim);
                }
            }
        }
    }

    Ok(claims)
}
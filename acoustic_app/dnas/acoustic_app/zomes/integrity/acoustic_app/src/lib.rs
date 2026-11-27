use hdi::prelude::*;

// Constants for physical laws
const SPEED_OF_SOUND: f64 = 343.0; // meters / second
const CLOCK_DRIFT_BUFFER: f64 = 0.050; // 50 milliseconds

/// An individual sensor's reading
#[derive(Clone, Debug, Serialize, Deserialize, SerializedBytes)]
pub struct SensorObservation {
    pub sensor_id: String,
    pub x_coord: f64,
    pub y_coord: f64,
    pub t_arrival: f64, // System epoch time in seconds
}

/// The bundled claim containing all observations for a single gunshot.
#[hdk_entry_helper]
pub struct AcousticTriangulationClaim {
    pub observations: Vec<SensorObservation>,
}

// Entry type registration for HDI 0.6.x
#[hdk_entry_types]
#[unit_enum(UnitEntryTypes)]
pub enum EntryTypes {
    AcousticTriangulationClaim(AcousticTriangulationClaim),
}
#[hdk_link_types]
pub enum LinkTypes {
    AllClaimsAnchor,
}

/// 100% Deterministic Validator.
#[hdk_extern]
pub fn validate(op: Op) -> ExternResult<ValidateCallbackResult> {
    match op {
        Op::StoreEntry(store_entry) => {
            // Direct conversion from Entry reference
            if let Ok(claim) = AcousticTriangulationClaim::try_from(&store_entry.entry) {
                
                // 1. Structural Validation: We need at least 3 sensors to triangulate
                if claim.observations.len() < 3 {
                    return Ok(ValidateCallbackResult::Invalid(
                        "Validation failed: At least 3 sensor observations are required to triangulate.".to_string()
                    ));
                }

                // 2. Physical Validation: Check every pair of sensors for speed-of-sound violations
                for i in 0..claim.observations.len() {
                    for j in (i + 1)..claim.observations.len() {
                        let obs_a = &claim.observations[i];
                        let obs_b = &claim.observations[j];

                        // Skip self-comparison
                        if obs_a.sensor_id == obs_b.sensor_id {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Validation failed: Duplicate sensor IDs detected in observations.".to_string()
                            ));
                        }

                        // Ensure all coordinates and timestamps are finite numbers (prevents NaN comparison bypass)
                        if !obs_a.x_coord.is_finite() || !obs_a.y_coord.is_finite() || !obs_a.t_arrival.is_finite()
                            || !obs_b.x_coord.is_finite() || !obs_b.y_coord.is_finite() || !obs_b.t_arrival.is_finite() {
                            return Ok(ValidateCallbackResult::Invalid(
                                "Validation failed: Non-finite coordinate or timestamp detected.".to_string()
                            ));
                        }

                        // Calculate physical distance: d = sqrt((dx)^2 + (dy)^2)
                        let dx = obs_a.x_coord - obs_b.x_coord;
                        let dy = obs_a.y_coord - obs_b.y_coord;
                        let distance = (dx * dx + dy * dy).sqrt();

                        // Sound travel limits
                        let max_travel_time = distance / SPEED_OF_SOUND;
                        let actual_time_delta = (obs_a.t_arrival - obs_b.t_arrival).abs();

                        if actual_time_delta > (max_travel_time + CLOCK_DRIFT_BUFFER) {
                            return Ok(ValidateCallbackResult::Invalid(format!(
                                "Physical validation failed. Speed of sound limits violated between {} and {}.",
                                obs_a.sensor_id, obs_b.sensor_id
                            )));
                        }
                    }
                }
                return Ok(ValidateCallbackResult::Valid);
            }
            Ok(ValidateCallbackResult::Valid)
        }
        _ => Ok(ValidateCallbackResult::Valid),
    }
}
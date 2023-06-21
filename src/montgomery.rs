// -*- mode: rust; -*-
//
// This file is part of curve25519-dalek.
// Copyright (c) 2016-2021 isis lovecruft
// Copyright (c) 2016-2019 Henry de Valence
// See LICENSE for licensing information.
//
// Authors:
// - isis agora lovecruft <isis@patternsinthevoid.net>
// - Henry de Valence <hdevalence@hdevalence.ca>

//! Scalar multiplication on the Montgomery form of Curve25519.
//!
//! To avoid notational confusion with the Edwards code, we use
//! variables \\( u, v \\) for the Montgomery curve, so that “Montgomery
//! \\(u\\)” here corresponds to “Montgomery \\(x\\)” elsewhere.
//!
//! Montgomery arithmetic works not on the curve itself, but on the
//! \\(u\\)-line, which discards sign information and unifies the curve
//! and its quadratic twist.  See [_Montgomery curves and their
//! arithmetic_][costello-smith] by Costello and Smith for more details.
//!
//! The `MontgomeryPoint` struct contains the affine \\(u\\)-coordinate
//! \\(u\_0(P)\\) of a point \\(P\\) on either the curve or the twist.
//! Here the map \\(u\_0 : \mathcal M \rightarrow \mathbb F\_p \\) is
//! defined by \\(u\_0((u,v)) = u\\); \\(u\_0(\mathcal O) = 0\\).  See
//! section 5.4 of Costello-Smith for more details.
//!
//! # Scalar Multiplication
//!
//! Scalar multiplication on `MontgomeryPoint`s is provided by the `*`
//! operator, which implements the Montgomery ladder.
//!
//! # Edwards Conversion
//!
//! The \\(2\\)-to-\\(1\\) map from the Edwards model to the Montgomery
//! \\(u\\)-line is provided by `EdwardsPoint::to_montgomery()`.
//!
//! To lift a `MontgomeryPoint` to an `EdwardsPoint`, use
//! `MontgomeryPoint::to_edwards()`, which takes a sign parameter.
//! This function rejects `MontgomeryPoints` which correspond to points
//! on the twist.
//!
//! [costello-smith]: https://eprint.iacr.org/2017/212.pdf

// We allow non snake_case names because coordinates in projective space are
// traditionally denoted by the capitalisation of their respective
// counterparts in affine space.  Yeah, you heard me, rustc, I'm gonna have my
// affine and projective cakes and eat both of them too.
#![allow(non_snake_case)]

use core::{
    hash::{Hash, Hasher},
    ops::{Mul, MulAssign},
};

use crate::constants::{APLUS2_OVER_FOUR, MONTGOMERY_A, MONTGOMERY_A_NEG, self};
use crate::edwards::{CompressedEdwardsY, EdwardsPoint};
use crate::field::FieldElement;
use crate::scalar::{clamp_integer, Scalar};

use crate::traits::Identity;

use subtle::Choice;
use subtle::ConstantTimeEq;
use subtle::{ConditionallyNegatable, ConditionallySelectable};

#[cfg(feature = "zeroize")]
use zeroize::Zeroize;


#[cfg(any(test, feature = "rand_core"))]
use rand_core::CryptoRngCore;

/// Holds the \\(u\\)-coordinate of a point on the Montgomery form of
/// Curve25519 or its twist.
#[derive(Copy, Clone, Debug, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MontgomeryPoint(pub [u8; 32]);

/// Equality of `MontgomeryPoint`s is defined mod p.
impl ConstantTimeEq for MontgomeryPoint {
    fn ct_eq(&self, other: &MontgomeryPoint) -> Choice {
        let self_fe = FieldElement::from_bytes(&self.0);
        let other_fe = FieldElement::from_bytes(&other.0);

        self_fe.ct_eq(&other_fe)
    }
}

impl PartialEq for MontgomeryPoint {
    fn eq(&self, other: &MontgomeryPoint) -> bool {
        self.ct_eq(other).into()
    }
}

impl Eq for MontgomeryPoint {}

// Equal MontgomeryPoints must hash to the same value. So we have to get them into a canonical
// encoding first
impl Hash for MontgomeryPoint {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Do a round trip through a `FieldElement`. `as_bytes` is guaranteed to give a canonical
        // 32-byte encoding
        let canonical_bytes = FieldElement::from_bytes(&self.0).as_bytes();
        canonical_bytes.hash(state);
    }
}

impl Identity for MontgomeryPoint {
    /// Return the group identity element, which has order 4.
    fn identity() -> MontgomeryPoint {
        MontgomeryPoint([0u8; 32])
    }
}

#[cfg(feature = "zeroize")]
impl Zeroize for MontgomeryPoint {
    fn zeroize(&mut self) {
        self.0.zeroize();
    }
}

impl MontgomeryPoint {
    /// Fixed-base scalar multiplication (i.e. multiplication by the base point).
    pub fn mul_base(scalar: &Scalar) -> Self {
        EdwardsPoint::mul_base(scalar).to_montgomery()
    }

    /// Multiply this point by `clamp_integer(bytes)`. For a description of clamping, see
    /// [`clamp_integer`].
    pub fn mul_clamped(self, bytes: [u8; 32]) -> Self {
        // We have to construct a Scalar that is not reduced mod l, which breaks scalar invariant
        // #2. But #2 is not necessary for correctness of variable-base multiplication. All that
        // needs to hold is invariant #1, i.e., the scalar is less than 2^255. This is guaranteed
        // by clamping.
        // Further, we don't do any reduction or arithmetic with this clamped value, so there's no
        // issues arising from the fact that the curve point is not necessarily in the prime-order
        // subgroup.
        let s = Scalar {
            bytes: clamp_integer(bytes),
        };
        s * self
    }

    /// Multiply the basepoint by `clamp_integer(bytes)`. For a description of clamping, see
    /// [`clamp_integer`].
    pub fn mul_base_clamped(bytes: [u8; 32]) -> Self {
        // See reasoning in Self::mul_clamped why it is OK to make an unreduced Scalar here. We
        // note that fixed-base multiplication is also defined for all values of `bytes` less than
        // 2^255.
        let s = Scalar {
            bytes: clamp_integer(bytes),
        };
        Self::mul_base(&s)
    }

    /// Generate a Montgomery point spanning the whole curve, useful for Elligator2 encoding
    /// Implemented by creating a normal prime-order point and adding a small torsion factor.
    pub fn generate_ephemeral_elligator(bytes: [u8; 32]) -> Self {
        let high_p = EdwardsPoint::mul_base_clamped(bytes);
        let low_p = constants::EIGHT_TORSION[(bytes[0] & 0b0000_0111) as usize];
        let point = high_p + low_p;
        point.to_montgomery()
    }

    /// Tries to generate an elligator point that also has a representative.
    /// This method has randomic run-time.
    #[cfg(any(test, feature = "rand_core"))]
    pub fn generate_ephemeral_elligator_random<R: CryptoRngCore + ?Sized>(rng: &mut R) -> ([u8; 32], [u8; 32]) {
        loop {
            let mut rng_data = [0u8; 33];
            rng.fill_bytes(&mut rng_data);
            let tweak = rng_data[32];
            let private = rng_data[..32].try_into().unwrap();

            let point = Self::generate_ephemeral_elligator(private);
            if let Some(repr) = point.to_elligator_representative(tweak) {
                return (private, repr);
            }
        }
    }

    /// View this `MontgomeryPoint` as an array of bytes.
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Convert this `MontgomeryPoint` to an array of bytes.
    pub const fn to_bytes(&self) -> [u8; 32] {
        self.0
    }

    /// Attempt to convert to an `EdwardsPoint`, using the supplied
    /// choice of sign for the `EdwardsPoint`.
    ///
    /// # Inputs
    ///
    /// * `sign`: a `u8` donating the desired sign of the resulting
    ///   `EdwardsPoint`.  `0` denotes positive and `1` negative.
    ///
    /// # Return
    ///
    /// * `Some(EdwardsPoint)` if `self` is the \\(u\\)-coordinate of a
    /// point on (the Montgomery form of) Curve25519;
    ///
    /// * `None` if `self` is the \\(u\\)-coordinate of a point on the
    /// twist of (the Montgomery form of) Curve25519;
    ///
    pub fn to_edwards(&self, sign: u8) -> Option<EdwardsPoint> {
        // To decompress the Montgomery u coordinate to an
        // `EdwardsPoint`, we apply the birational map to obtain the
        // Edwards y coordinate, then do Edwards decompression.
        //
        // The birational map is y = (u-1)/(u+1).
        //
        // The exceptional points are the zeros of the denominator,
        // i.e., u = -1.
        //
        // But when u = -1, v^2 = u*(u^2+486662*u+1) = 486660.
        //
        // Since this is nonsquare mod p, u = -1 corresponds to a point
        // on the twist, not the curve, so we can reject it early.

        let u = FieldElement::from_bytes(&self.0);

        if u == FieldElement::MINUS_ONE {
            return None;
        }

        let one = FieldElement::ONE;

        let y = &(&u - &one) * &(&u + &one).invert();

        let mut y_bytes = y.as_bytes();
        y_bytes[31] ^= sign << 7;

        CompressedEdwardsY(y_bytes).decompress()
    }

    /// Perform the Elligator2 mapping to a Montgomery point.
    ///
    /// This will also ignore the higher 2 bits, not used by the encoding.
    /// See <https://elligator.org/>
    ///
    /// # Inputs
    ///
    /// * `data`: The bitstream that represents the returned point in the Elligator2 mappng.
    ///
    /// # Return
    ///
    /// * `MontgomeryPoint` The point represented by `data`.
    pub fn from_elligator_representative(data: &[u8; 32]) -> MontgomeryPoint {
        let mut data = *data;
        data[31] &= 0b0011_1111;
        let r_0 = FieldElement::from_bytes(&data);
        elligator_encode(&r_0)
    }

    /// Perform the Elligator2 mapping to a bitstream representative.
    ///
    /// Reverse of [from_elligator_representative](MontgomeryPoint::from_elligator_representative),
    /// see <https://elligator.org/>.
    ///
    /// # Inputs
    ///
    /// * `tweak`: a `u8` used to randomize the elemnt given as result
    ///           it will determine wether the element is positive/negative
    ///           and the higher 2 bits (unused by the encoding)
    ///
    /// # Return
    ///
    /// * `Some([u8; 32])` if `self` has a Elligator2 representative (which one is decided by `tweak`)
    ///
    /// * `None` if `self` cannot be derived from an Elligator2 representative.
    pub fn to_elligator_representative(&self, tweak: u8) -> Option<[u8; 32]> {
        //let one = FieldElement::ONE;
        let u = FieldElement::from_bytes(self.as_bytes());
        let u_plus_A = &u + &MONTGOMERY_A;
        let uu_plus_uA = &u * &u_plus_A;

        // Condition: u != -A
        if u == MONTGOMERY_A_NEG {
            return None
        }

        // Condition: -2u(u+A) is a square
        let uu2_plus_uA2 = &uu_plus_uA + &uu_plus_uA;
        // We compute root = sqrt(-1/2u(u+A)) to speed up the calculation.
        // This is a square if and only if -2u(u+A) is.
        let (is_square, root) = FieldElement::sqrt_ratio_i(&FieldElement::ONE,
                                                           &-&uu2_plus_uA2);
        if !bool::from(is_square | root.is_zero()) {
            return None;
        }

        // if !v_is_negative: r = sqrt(-u / 2(u + a)) = root * u
        // if  v_is_negative: r = sqrt(-(u+A) / 2u)   = root * (u + A)
        let add = FieldElement::conditional_select(&u, &u_plus_A, (tweak & 1).into());
        let r = &root * &add;

        // Both r and -r are valid results. Pick the nonnegative one.
        // FieldElement::is_negative and is_negative used in elligator reference implementation differ!
        // FieldElement checks the low bit, while elligator checks that r is in [p.+1 / 2.. p-1]
        // since we want to follow the reference implementation, we'll use the second definition
        let is_negative = (&r + &r).is_negative();
        let result = FieldElement::conditional_select(&r, &-&r, is_negative);
        let mut data = result.as_bytes();

        assert_eq!(data[31] & 0b1100_0000, 0);
        data[31] |= tweak & 0b1100_0000;

        return Some(data);
    }

}

/// Perform the Elligator2 mapping to a Montgomery point.
///
/// See <https://tools.ietf.org/html/draft-irtf-cfrg-hash-to-curve-10#section-6.7.1>
//
// TODO Determine how much of the hash-to-group API should be exposed after the CFRG
//      draft gets into a more polished/accepted state.
#[allow(unused)]
pub(crate) fn elligator_encode(r_0: &FieldElement) -> MontgomeryPoint {
    let one = FieldElement::ONE;
    let d_1 = &one + &r_0.square2(); /* 2r^2 */

    let d = &MONTGOMERY_A_NEG * &(d_1.invert()); /* -A/(1+2r^2) */

    let d_sq = &d.square();
    let au = &MONTGOMERY_A * &d;

    let inner = &(d_sq + &au) + &one;
    let eps = &d * &inner; /* eps = d^3 + Ad^2 + d */

    let (eps_is_sq, _eps) = FieldElement::sqrt_ratio_i(&eps, &one);

    let zero = FieldElement::ZERO;
    let Atemp = FieldElement::conditional_select(&MONTGOMERY_A, &zero, eps_is_sq); /* 0, or A if nonsquare*/
    let mut u = &d + &Atemp; /* d, or d+A if nonsquare */
    u.conditional_negate(!eps_is_sq); /* d, or -d-A if nonsquare */

    MontgomeryPoint(u.as_bytes())
}

/// A `ProjectivePoint` holds a point on the projective line
/// \\( \mathbb P(\mathbb F\_p) \\), which we identify with the Kummer
/// line of the Montgomery curve.
#[derive(Copy, Clone, Debug)]
struct ProjectivePoint {
    pub U: FieldElement,
    pub W: FieldElement,
}

impl Identity for ProjectivePoint {
    fn identity() -> ProjectivePoint {
        ProjectivePoint {
            U: FieldElement::ONE,
            W: FieldElement::ZERO,
        }
    }
}

impl Default for ProjectivePoint {
    fn default() -> ProjectivePoint {
        ProjectivePoint::identity()
    }
}

impl ConditionallySelectable for ProjectivePoint {
    fn conditional_select(
        a: &ProjectivePoint,
        b: &ProjectivePoint,
        choice: Choice,
    ) -> ProjectivePoint {
        ProjectivePoint {
            U: FieldElement::conditional_select(&a.U, &b.U, choice),
            W: FieldElement::conditional_select(&a.W, &b.W, choice),
        }
    }
}

impl ProjectivePoint {
    /// Dehomogenize this point to affine coordinates.
    ///
    /// # Return
    ///
    /// * \\( u = U / W \\) if \\( W \neq 0 \\);
    /// * \\( 0 \\) if \\( W \eq 0 \\);
    pub fn as_affine(&self) -> MontgomeryPoint {
        let u = &self.U * &self.W.invert();
        MontgomeryPoint(u.as_bytes())
    }
}

/// Perform the double-and-add step of the Montgomery ladder.
///
/// Given projective points
/// \\( (U\_P : W\_P) = u(P) \\),
/// \\( (U\_Q : W\_Q) = u(Q) \\),
/// and the affine difference
/// \\(      u\_{P-Q} = u(P-Q) \\), set
/// $$
///     (U\_P : W\_P) \gets u(\[2\]P)
/// $$
/// and
/// $$
///     (U\_Q : W\_Q) \gets u(P + Q).
/// $$
#[rustfmt::skip] // keep alignment of explanatory comments
fn differential_add_and_double(
    P: &mut ProjectivePoint,
    Q: &mut ProjectivePoint,
    affine_PmQ: &FieldElement,
) {
    let t0 = &P.U + &P.W;
    let t1 = &P.U - &P.W;
    let t2 = &Q.U + &Q.W;
    let t3 = &Q.U - &Q.W;

    let t4 = t0.square();   // (U_P + W_P)^2 = U_P^2 + 2 U_P W_P + W_P^2
    let t5 = t1.square();   // (U_P - W_P)^2 = U_P^2 - 2 U_P W_P + W_P^2

    let t6 = &t4 - &t5;     // 4 U_P W_P

    let t7 = &t0 * &t3;     // (U_P + W_P) (U_Q - W_Q) = U_P U_Q + W_P U_Q - U_P W_Q - W_P W_Q
    let t8 = &t1 * &t2;     // (U_P - W_P) (U_Q + W_Q) = U_P U_Q - W_P U_Q + U_P W_Q - W_P W_Q

    let t9  = &t7 + &t8;    // 2 (U_P U_Q - W_P W_Q)
    let t10 = &t7 - &t8;    // 2 (W_P U_Q - U_P W_Q)

    let t11 =  t9.square(); // 4 (U_P U_Q - W_P W_Q)^2
    let t12 = t10.square(); // 4 (W_P U_Q - U_P W_Q)^2

    let t13 = &APLUS2_OVER_FOUR * &t6; // (A + 2) U_P U_Q

    let t14 = &t4 * &t5;    // ((U_P + W_P)(U_P - W_P))^2 = (U_P^2 - W_P^2)^2
    let t15 = &t13 + &t5;   // (U_P - W_P)^2 + (A + 2) U_P W_P

    let t16 = &t6 * &t15;   // 4 (U_P W_P) ((U_P - W_P)^2 + (A + 2) U_P W_P)

    let t17 = affine_PmQ * &t12; // U_D * 4 (W_P U_Q - U_P W_Q)^2
    let t18 = t11;               // W_D * 4 (U_P U_Q - W_P W_Q)^2

    P.U = t14;  // U_{P'} = (U_P + W_P)^2 (U_P - W_P)^2
    P.W = t16;  // W_{P'} = (4 U_P W_P) ((U_P - W_P)^2 + ((A + 2)/4) 4 U_P W_P)
    Q.U = t18;  // U_{Q'} = W_D * 4 (U_P U_Q - W_P W_Q)^2
    Q.W = t17;  // W_{Q'} = U_D * 4 (W_P U_Q - U_P W_Q)^2
}

define_mul_assign_variants!(LHS = MontgomeryPoint, RHS = Scalar);

define_mul_variants!(
    LHS = MontgomeryPoint,
    RHS = Scalar,
    Output = MontgomeryPoint
);
define_mul_variants!(
    LHS = Scalar,
    RHS = MontgomeryPoint,
    Output = MontgomeryPoint
);

/// Multiply this `MontgomeryPoint` by a `Scalar`.
impl<'a, 'b> Mul<&'b Scalar> for &'a MontgomeryPoint {
    type Output = MontgomeryPoint;

    /// Given `self` \\( = u\_0(P) \\), and a `Scalar` \\(n\\), return \\( u\_0(\[n\]P) \\).
    fn mul(self, scalar: &'b Scalar) -> MontgomeryPoint {
        // Algorithm 8 of Costello-Smith 2017
        let affine_u = FieldElement::from_bytes(&self.0);
        let mut x0 = ProjectivePoint::identity();
        let mut x1 = ProjectivePoint {
            U: affine_u,
            W: FieldElement::ONE,
        };

        // NOTE: The below swap-double-add routine skips the first iteration, i.e., it assumes the
        // MSB of `scalar` is 0. This is allowed, since it follows from Scalar invariant #1.

        // Go through the bits from most to least significant, using a sliding window of 2
        let mut bits = scalar.bits_le().rev();
        let mut prev_bit = bits.next().unwrap();
        for cur_bit in bits {
            let choice: u8 = (prev_bit ^ cur_bit) as u8;

            debug_assert!(choice == 0 || choice == 1);

            ProjectivePoint::conditional_swap(&mut x0, &mut x1, choice.into());
            differential_add_and_double(&mut x0, &mut x1, &affine_u);

            prev_bit = cur_bit;
        }
        // The final value of prev_bit above is scalar.bits()[0], i.e., the LSB of scalar
        ProjectivePoint::conditional_swap(&mut x0, &mut x1, Choice::from(prev_bit as u8));
        // Don't leave the bit in the stack
        #[cfg(feature = "zeroize")]
        prev_bit.zeroize();

        x0.as_affine()
    }
}

impl<'b> MulAssign<&'b Scalar> for MontgomeryPoint {
    fn mul_assign(&mut self, scalar: &'b Scalar) {
        *self = (self as &MontgomeryPoint) * scalar;
    }
}

impl<'a, 'b> Mul<&'b MontgomeryPoint> for &'a Scalar {
    type Output = MontgomeryPoint;

    fn mul(self, point: &'b MontgomeryPoint) -> MontgomeryPoint {
        point * self
    }
}

// ------------------------------------------------------------------------
// Tests
// ------------------------------------------------------------------------

#[cfg(test)]
mod test {
    use std::{path::PathBuf, fs};

    use super::*;
    use crate::constants;

    #[cfg(feature = "alloc")]
    use alloc::vec::Vec;

    use rand::Rng;
    use rand_core::{RngCore, OsRng};

    #[test]
    fn identity_in_different_coordinates() {
        let id_projective = ProjectivePoint::identity();
        let id_montgomery = id_projective.as_affine();

        assert!(id_montgomery == MontgomeryPoint::identity());
    }

    #[test]
    fn identity_in_different_models() {
        assert!(EdwardsPoint::identity().to_montgomery() == MontgomeryPoint::identity());
    }

    #[test]
    fn montgomery_elligator_encode_decode() {
        for _i in 0..4096 {
            let mut bits = [0u8; 32];
            OsRng.fill_bytes(&mut bits);
            // Remove padding:
            bits[31] &= 0b0011_1111;

            let eg = MontgomeryPoint::from_elligator_representative(&bits);

            // Up to four different field values may encode to a
            // single MontgomeryPoint. Firstly, the MontgomeryPoint
            // loses the v-coordinate, so it may represent two
            // different curve points, (u, v) and (u, -v). The
            // elligator_decode function is given a sign argument to
            // indicate which one is meant.
            //
            // Second, for each curve point (except zero), two
            // distinct field elements encode to it, r and -r. The
            // elligator_decode function returns the nonnegative one.
            //
            // We check here that one of these four values is equal to
            // the original input to elligator_encode.

            let mut found = false;

            for i in 0..=1 {
                let decoded = eg.to_elligator_representative(i)
                    .expect("Elligator decode failed");
                let fe = FieldElement::from_bytes(&bits);
                let dec = FieldElement::from_bytes(&decoded);
                for j in &[fe, -&fe] {
                    found |= dec == *j;
                }
            }

            assert!(found);
        }
    }

    #[test]
    fn montgomery_elligator_decode_encode() {
        let mut csprng = OsRng;
        for _i in 0..100 {
            let s: Scalar = Scalar::random(&mut csprng);
            let p_edwards = EdwardsPoint::mul_base(&s);
            let p_montgomery: MontgomeryPoint = p_edwards.to_montgomery();
            let tweak: u8 = csprng.gen();

            let result = p_montgomery.to_elligator_representative(tweak);

            if let Some(fe) = result {
                let encoded = MontgomeryPoint::from_elligator_representative(&fe);

                assert_eq!(encoded, p_montgomery);
            }
        }
    }

    #[test]
    fn montgomery_elligator_lifecycle() {
        let mut rng: rand::rngs::ThreadRng = rand::thread_rng();
        for _i in 0..10 {
            let (a_priv_key, a_repr) = MontgomeryPoint::generate_ephemeral_elligator_random(&mut rng);
            let (b_priv_key, b_repr) = MontgomeryPoint::generate_ephemeral_elligator_random(&mut rng);

            // A: b.send(a_repr)
            // B: a.send(b_repr)
            let a_secret = MontgomeryPoint::from_elligator_representative(&b_repr).mul_clamped(a_priv_key);
            let b_secret = MontgomeryPoint::from_elligator_representative(&a_repr).mul_clamped(b_priv_key);

            assert_eq!(a_secret, b_secret);
            eprint!(".");
        }
    }

    #[test]
    #[cfg(feature = "serde")]
    fn serde_bincode_basepoint_roundtrip() {
        use bincode;

        let encoded = bincode::serialize(&constants::X25519_BASEPOINT).unwrap();
        let decoded: MontgomeryPoint = bincode::deserialize(&encoded).unwrap();

        assert_eq!(encoded.len(), 32);
        assert_eq!(decoded, constants::X25519_BASEPOINT);

        let raw_bytes = constants::X25519_BASEPOINT.as_bytes();
        let bp: MontgomeryPoint = bincode::deserialize(raw_bytes).unwrap();
        assert_eq!(bp, constants::X25519_BASEPOINT);
    }

    /// Test Montgomery -> Edwards on the X/Ed25519 basepoint
    #[test]
    fn basepoint_montgomery_to_edwards() {
        // sign bit = 0 => basepoint
        assert_eq!(
            constants::ED25519_BASEPOINT_POINT,
            constants::X25519_BASEPOINT.to_edwards(0).unwrap()
        );
        // sign bit = 1 => minus basepoint
        assert_eq!(
            -constants::ED25519_BASEPOINT_POINT,
            constants::X25519_BASEPOINT.to_edwards(1).unwrap()
        );
    }

    /// Test Edwards -> Montgomery on the X/Ed25519 basepoint
    #[test]
    fn basepoint_edwards_to_montgomery() {
        assert_eq!(
            constants::ED25519_BASEPOINT_POINT.to_montgomery(),
            constants::X25519_BASEPOINT
        );
    }

    /// Check that Montgomery -> Edwards fails for points on the twist.
    #[test]
    fn montgomery_to_edwards_rejects_twist() {
        let one = FieldElement::ONE;

        // u = 2 corresponds to a point on the twist.
        let two = MontgomeryPoint((&one + &one).as_bytes());

        assert!(two.to_edwards(0).is_none());

        // u = -1 corresponds to a point on the twist, but should be
        // checked explicitly because it's an exceptional point for the
        // birational map.  For instance, libsignal will accept it.
        let minus_one = MontgomeryPoint((-&one).as_bytes());

        assert!(minus_one.to_edwards(0).is_none());
    }

    #[test]
    fn eq_defined_mod_p() {
        let mut u18_bytes = [0u8; 32];
        u18_bytes[0] = 18;
        let u18 = MontgomeryPoint(u18_bytes);
        let u18_unred = MontgomeryPoint([255; 32]);

        assert_eq!(u18, u18_unred);
    }

    #[test]
    fn montgomery_ladder_matches_edwards_scalarmult() {
        let mut csprng = rand_core::OsRng;

        for _ in 0..100 {
            let s: Scalar = Scalar::random(&mut csprng);
            let p_edwards = EdwardsPoint::mul_base(&s);
            let p_montgomery: MontgomeryPoint = p_edwards.to_montgomery();

            let expected = s * p_edwards;
            let result = s * p_montgomery;

            assert_eq!(result, expected.to_montgomery())
        }
    }

    /// Check that mul_base_clamped and mul_clamped agree
    #[test]
    fn mul_base_clamped() {
        let mut csprng = rand_core::OsRng;

        // Test agreement on a large integer. Even after clamping, this is not reduced mod l.
        let a_bytes = [0xff; 32];
        assert_eq!(
            MontgomeryPoint::mul_base_clamped(a_bytes),
            constants::X25519_BASEPOINT.mul_clamped(a_bytes)
        );

        // Test agreement on random integers
        for _ in 0..100 {
            // This will be reduced mod l with probability l / 2^256 ≈ 6.25%
            let mut a_bytes = [0u8; 32];
            csprng.fill_bytes(&mut a_bytes);

            assert_eq!(
                MontgomeryPoint::mul_base_clamped(a_bytes),
                constants::X25519_BASEPOINT.mul_clamped(a_bytes)
            );
        }
    }

    #[cfg(feature = "alloc")]
    const ELLIGATOR_CORRECT_OUTPUT: [u8; 32] = [
        0x5f, 0x35, 0x20, 0x00, 0x1c, 0x6c, 0x99, 0x36, 0xa3, 0x12, 0x06, 0xaf, 0xe7, 0xc7, 0xac,
        0x22, 0x4e, 0x88, 0x61, 0x61, 0x9b, 0xf9, 0x88, 0x72, 0x44, 0x49, 0x15, 0x89, 0x9d, 0x95,
        0xf4, 0x6e,
    ];

    #[test]
    #[cfg(feature = "alloc")]
    fn montgomery_elligator_correct() {
        let bytes: Vec<u8> = (0u8..32u8).collect();
        let bits_in: [u8; 32] = (&bytes[..]).try_into().expect("Range invariant broken");

        let fe = FieldElement::from_bytes(&bits_in);
        let eg = elligator_encode(&fe);
        assert_eq!(eg.to_bytes(), ELLIGATOR_CORRECT_OUTPUT);
    }

    #[test]
    fn montgomery_elligator_zero_zero() {
        let zero = [0u8; 32];
        let fe = FieldElement::from_bytes(&zero);
        let eg = elligator_encode(&fe);
        assert_eq!(eg.to_bytes(), zero);
    }

    #[test]
    fn montgomery_elligator_test_vec() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("vendor/curve25519_direct.vec");
        let test_file = fs::read_to_string(path).expect("Failed reading test vec");
        let test_lines = test_file.split('\n');
        let mut test_lines_it = test_lines.into_iter();
        let mut reprs = vec![];
        let mut u_coords = vec![];
        loop {
            let (repr, u, v, empty) = (test_lines_it.next(), test_lines_it.next(), test_lines_it.next(), test_lines_it.next());
            if let (Some(reprt), Some(ut), _, _) = (repr, u, v, empty) {
                let mut r = [0u8; 32];
                let mut u = [0u8; 32];
                hex::decode_to_slice(reprt.strip_suffix(":").unwrap(), &mut r).expect("Invalid repr format");
                hex::decode_to_slice(ut.strip_suffix(":").unwrap(), &mut u).expect("Invalid u-cord format");
                reprs.push(r);
                u_coords.push(u);
            } else {
                break
            }
        }

        for (r, u) in reprs.iter().zip(u_coords.iter()) {
            let fe = FieldElement::from_bytes(&r);
            let eg = elligator_encode(&fe);
            assert_eq!(eg.to_bytes(), *u);
        }
    }

    #[test]
    fn montgomery_elligator_inv_test_vec() {
        let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        path.push("vendor/curve25519_inverse.vec");
        let test_file = fs::read_to_string(path).expect("Failed reading test vec");
        let test_lines = test_file.split('\n');
        let mut test_lines_it = test_lines.into_iter();
        let mut u_coords = vec![];
        let mut signs = vec![];
        let mut reprs = vec![];
        loop {
            let (u, sign, _has_repr, repr, _empty) = (test_lines_it.next(), test_lines_it.next(), test_lines_it.next(), test_lines_it.next(), test_lines_it.next());
            if let (Some(ut), Some(signt), Some(reprt)) = (u, sign, repr) {

                let mut u = [0u8; 32];
                let sign = signt == "01:";

                hex::decode_to_slice(ut.strip_suffix(":").unwrap(), &mut u).expect("Invalid u-cord format");
                let r = if reprt.len() == 1 {
                    None
                } else {
                    let mut r = [0u8; 32];
                    hex::decode_to_slice(reprt.strip_suffix(":").unwrap(), &mut r).expect("Invalid repr format");
                    Some(r)
                };
                u_coords.push(u);
                signs.push(sign);
                reprs.push(r);
            } else {
                break
            }
        }

        for ((u, sign), r) in u_coords.iter().zip(signs.iter()).zip(reprs.iter()) {
            let data = MontgomeryPoint(*u).to_elligator_representative(*sign as u8);

            assert_eq!(data, *r);
        }
    }
}

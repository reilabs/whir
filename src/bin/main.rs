use std::borrow::Borrow;
use std::time::Instant;

use ark_crypto_primitives::merkle_tree::IdentityDigestConverter;
use ark_crypto_primitives::{
    crh::{CRHScheme, TwoToOneCRHScheme},
    merkle_tree::Config,
    Error,
};
use ark_ff::{FftField, Field};
use ark_serialize::CanonicalSerialize;
use nimue::{Arthur, DefaultHash, IOPattern, Merlin, ProofResult};
use whir::{
    cmdline_utils::{AvailableFields, AvailableMerkle, WhirType},
    crypto::{
        fields,
        merkle_tree::{self, HashCounter},
    },
    parameters::*,
    poly_utils::{coeffs::CoefficientList, MultilinearPoint},
    whir::Statement,
};

use nimue_pow::blake3::Blake3PoW;

use clap::Parser;
use nimue::hash::sponge::{DuplexSponge, Sponge};
use nimue::plugins::ark::{FieldIOPattern, FieldReader, FieldWriter};
use rand::Rng;
use whir::crypto::fields::Field256;
use whir::skyscraper::{Skyscraper, SkyscraperSponge};
use whir::whir::fs_utils::{DigestReader, DigestWriter};
use whir::whir::iopattern::DigestIOPattern;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short = 't', long = "type", default_value = "PCS")]
    protocol_type: WhirType,

    #[arg(short = 'l', long, default_value = "100")]
    security_level: usize,

    #[arg(short = 'p', long)]
    pow_bits: Option<usize>,

    #[arg(short = 'd', long, default_value = "20")]
    num_variables: usize,

    #[arg(short = 'e', long = "evaluations", default_value = "1")]
    num_evaluations: usize,

    #[arg(short = 'r', long, default_value = "1")]
    rate: usize,

    #[arg(long = "reps", default_value = "1000")]
    verifier_repetitions: usize,

    #[arg(short = 'k', long = "fold", default_value = "4")]
    folding_factor: usize,

    #[arg(long = "sec", default_value = "ConjectureList")]
    soundness_type: SoundnessType,

    #[arg(long = "fold_type", default_value = "ProverHelps")]
    fold_optimisation: FoldType,

    #[arg(short = 'f', long = "field", default_value = "Goldilocks2")]
    field: AvailableFields,

    #[arg(long = "hash", default_value = "Blake3")]
    merkle_tree: AvailableMerkle,
}

type PowStrategy = Blake3PoW;

fn main() {
    let mut args = Args::parse();
    let field = args.field;
    let merkle = args.merkle_tree;

    if args.pow_bits.is_none() {
        args.pow_bits = Some(default_max_pow(args.num_variables, args.rate));
    }

    let mut rng = ark_std::test_rng();

    match (field, merkle) {
        (AvailableFields::Goldilocks1, AvailableMerkle::Blake3) => {
            use fields::Field64 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks1, AvailableMerkle::Keccak256) => {
            use fields::Field64 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks2, AvailableMerkle::Blake3) => {
            use fields::Field64_2 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks2, AvailableMerkle::Keccak256) => {
            use fields::Field64_2 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks3, AvailableMerkle::Blake3) => {
            use fields::Field64_3 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Goldilocks3, AvailableMerkle::Keccak256) => {
            use fields::Field64_3 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field128, AvailableMerkle::Blake3) => {
            use fields::Field128 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field128, AvailableMerkle::Keccak256) => {
            use fields::Field128 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field192, AvailableMerkle::Blake3) => {
            use fields::Field192 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field192, AvailableMerkle::Keccak256) => {
            use fields::Field192 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field256, AvailableMerkle::Blake3) => {
            use fields::Field256 as F;
            use merkle_tree::blake3 as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }

        (AvailableFields::Field256, AvailableMerkle::Keccak256) => {
            use fields::Field256 as F;
            use merkle_tree::keccak as mt;

            let (leaf_hash_params, two_to_one_params) = mt::default_config::<F>(&mut rng);
            run_whir::<F, mt::MerkleTreeParams<F>>(args, leaf_hash_params, two_to_one_params);
        }
    }
}

fn run_whir<F, MerkleConfig>(
    args: Args,
    leaf_hash_params: <<MerkleConfig as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_params: <<MerkleConfig as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) where
    F: FftField + CanonicalSerialize,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    IOPattern: DigestIOPattern<MerkleConfig>,
    Merlin: DigestWriter<MerkleConfig>, for<'a> Arthur<'a>: DigestReader<MerkleConfig>
{
    match args.protocol_type {
        WhirType::PCS => run_whir_pcs(args),
        WhirType::LDT => {
            run_whir_as_ldt::<F, MerkleConfig>(args, leaf_hash_params, two_to_one_params)
        }
    }
}

fn run_whir_as_ldt<F, MerkleConfig>(
    args: Args,
    leaf_hash_params: <<MerkleConfig as Config>::LeafHash as CRHScheme>::Parameters,
    two_to_one_params: <<MerkleConfig as Config>::TwoToOneHash as TwoToOneCRHScheme>::Parameters,
) where
    F: FftField + CanonicalSerialize,
    MerkleConfig: Config<Leaf = [F]> + Clone,
    MerkleConfig::InnerDigest: AsRef<[u8]> + From<[u8; 32]>,
    IOPattern: DigestIOPattern<MerkleConfig>,
    Merlin: DigestWriter<MerkleConfig>,
    for <'a> Arthur<'a>: DigestReader<MerkleConfig>,
{
    use whir::whir::{
        committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
        verifier::Verifier,
    };

    // Runs as a LDT
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let fold_optimisation = args.fold_optimisation;
    let soundness_type = args.soundness_type;

    if args.num_evaluations > 1 {
        println!("Warning: running as LDT but a number of evaluations to be proven was specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = WhirParameters::<MerkleConfig, PowStrategy> {
        initial_statement: false,
        security_level,
        pow_bits,
        folding_factor,
        leaf_hash_params,
        two_to_one_params,
        soundness_type,
        fold_optimisation,
        _pow_parameters: Default::default(),
        starting_log_inv_rate: starting_rate,
    };

    let params = WhirConfig::<F, MerkleConfig, PowStrategy>::new(mv_params, whir_params.clone());

    let io = IOPattern::<DefaultHash>::new("🌪️")
        .commit_statement(&params)
        .add_whir_proof(&params);

    let mut merlin = io.to_merlin();

    println!("=========================================");
    println!("Whir (LDT) 🌪️");
    println!("Field: {:?} and MT: {:?}", args.field, args.merkle_tree);
    println!("{}", params);
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    use ark_ff::Field;
    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );

    let whir_prover_time = Instant::now();

    let committer = Committer::new(params.clone());
    let witness = committer.commit(&mut merlin, polynomial).unwrap();

    let prover = Prover(params.clone());

    let proof = prover
        .prove(&mut merlin, Statement::default(), witness)
        .unwrap();

    dbg!(whir_prover_time.elapsed());

    // Serialize proof
    let transcript = merlin.transcript().to_vec();
    let mut proof_bytes = vec![];
    proof.serialize_compressed(&mut proof_bytes).unwrap();

    let proof_size = transcript.len() + proof_bytes.len();
    dbg!(proof_size);

    // Just not to count that initial inversion (which could be precomputed)
    let verifier = Verifier::new(params.clone());

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..reps {
        let mut arthur = io.to_arthur(&transcript);
        verifier
            .verify(&mut arthur, &Statement::default(), &proof)
            .unwrap();
    }
    dbg!(whir_verifier_time.elapsed() / reps as u32);
    dbg!(HashCounter::get() as f64 / reps as f64);
}
struct SkyscraperCRH;

impl CRHScheme for SkyscraperCRH {
    type Input = [Field256];
    type Output = Field256;
    type Parameters = ();

    fn setup<R: Rng>(r: &mut R) -> Result<Self::Parameters, Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        parameters: &Self::Parameters,
        input: T,
    ) -> Result<Self::Output, Error> {
        let elems = input.borrow();
        elems
            .iter()
            .cloned()
            .reduce(whir::skyscraper::compress)
            .ok_or(Error::IncorrectInputLength(0))
    }
}

struct SkyscraperTwoToOne;

impl TwoToOneCRHScheme for SkyscraperTwoToOne {
    type Input = Field256;
    type Output = Field256;
    type Parameters = ();

    fn setup<R: Rng>(r: &mut R) -> Result<Self::Parameters, Error> {
        Ok(())
    }

    fn evaluate<T: Borrow<Self::Input>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, Error> {
        Ok(whir::skyscraper::compress(
            left_input.borrow().clone(),
            right_input.borrow().clone(),
        ))
    }

    fn compress<T: Borrow<Self::Output>>(
        parameters: &Self::Parameters,
        left_input: T,
        right_input: T,
    ) -> Result<Self::Output, Error> {
        <Self as TwoToOneCRHScheme>::evaluate(parameters, left_input, right_input)
    }
}

#[derive(Clone, Copy)]
struct SkyscraperMerkleConfig;

impl Config for SkyscraperMerkleConfig {
    type Leaf = [Field256];
    type LeafDigest = Field256;
    type LeafInnerDigestConverter = IdentityDigestConverter<Field256>;
    type InnerDigest = Field256;
    type LeafHash = SkyscraperCRH;
    type TwoToOneHash = SkyscraperTwoToOne;
}

impl DigestIOPattern<SkyscraperMerkleConfig> for IOPattern<SkyscraperSponge, Field256> {
    fn add_digest(self, label: &str) -> Self {
        <Self as FieldIOPattern<Field256>>::add_scalars(self, 1, label)
    }
}

impl DigestWriter<SkyscraperMerkleConfig> for Merlin<SkyscraperSponge, Field256> {
    fn add_digest(&mut self, digest: Field256) -> ProofResult<()> {
        self.add_scalars(&[digest])
    }
}

impl <'a> DigestReader<SkyscraperMerkleConfig> for Arthur<'a, SkyscraperSponge, Field256> {
    fn read_digest(&mut self) -> ProofResult<Field256> {
        let [r] = self.next_scalars()?;
        Ok(r)
    }
}

fn run_whir_pcs(args: Args) {
    use fields::Field256 as F;
    use whir::whir::{
        committer::Committer, iopattern::WhirIOPattern, parameters::WhirConfig, prover::Prover,
        verifier::Verifier, whir_proof_size, Statement,
    };

    // Runs as a PCS
    let security_level = args.security_level;
    let pow_bits = args.pow_bits.unwrap();
    let num_variables = args.num_variables;
    let starting_rate = args.rate;
    let reps = args.verifier_repetitions;
    let folding_factor = args.folding_factor;
    let fold_optimisation = args.fold_optimisation;
    let soundness_type = args.soundness_type;
    let num_evaluations = args.num_evaluations;

    if num_evaluations == 0 {
        println!("Warning: running as PCS but no evaluations specified.");
    }

    let num_coeffs = 1 << num_variables;

    let mv_params = MultivariateParameters::<F>::new(num_variables);

    let whir_params = WhirParameters::<SkyscraperMerkleConfig, PowStrategy> {
        initial_statement: true,
        security_level,
        pow_bits,
        folding_factor,
        leaf_hash_params: (),
        two_to_one_params: (),
        soundness_type,
        fold_optimisation,
        _pow_parameters: Default::default(),
        starting_log_inv_rate: starting_rate,
    };

    let params = WhirConfig::<F, SkyscraperMerkleConfig, PowStrategy>::new(mv_params, whir_params);

    let io = IOPattern::<SkyscraperSponge, F>::new("🌪️")
        .commit_statement(&params)
        .add_whir_proof(&params);

    println!("io: {:?}", io);

    let mut merlin = io.to_merlin();

    println!("=========================================");
    println!("Whir (PCS) 🌪️");
    println!("Field: {:?} and MT: {:?}", args.field, args.merkle_tree);
    println!("{}", params);
    if !params.check_pow_bits() {
        println!("WARN: more PoW bits required than what specified.");
    }

    use ark_ff::Field;
    let polynomial = CoefficientList::new(
        (0..num_coeffs)
            .map(<F as Field>::BasePrimeField::from)
            .collect(),
    );
    let points: Vec<_> = (0..num_evaluations)
        .map(|i| MultilinearPoint(vec![F::from(i as u64); num_variables]))
        .collect();
    let evaluations = points
        .iter()
        .map(|point| polynomial.evaluate_at_extension(point))
        .collect();

    let statement = Statement {
        points,
        evaluations,
    };

    let whir_prover_time = Instant::now();

    let committer = Committer::new(params.clone());
    let witness = committer.commit(&mut merlin, polynomial).unwrap();

    let prover = Prover(params.clone());

    let proof = prover
        .prove(&mut merlin, statement.clone(), witness)
        .unwrap();

    println!("Prover time: {:.1?}", whir_prover_time.elapsed());
    println!(
        "Proof size: {:.1} KiB",
        whir_proof_size(merlin.transcript(), &proof) as f64 / 1024.0
    );

    // Just not to count that initial inversion (which could be precomputed)
    let verifier = Verifier::new(params);

    HashCounter::reset();
    let whir_verifier_time = Instant::now();
    for _ in 0..1 {
        let mut arthur = io.to_arthur(merlin.transcript());
        verifier.verify(&mut arthur, &statement, &proof).unwrap();
    }
    println!(
        "Verifier time: {:.1?}",
        whir_verifier_time.elapsed() / reps as u32
    );
    println!(
        "Average hashes: {:.1}k",
        (HashCounter::get() as f64 / reps as f64) / 1000.0
    );
}

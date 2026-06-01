use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::model::{build_factor_activation, factor_activation_for_row_int};
use crate::types::InducingFactor;

// ── CPD and BN types ───────────────────────────────────────────

/// A Conditional Probability Distribution (CPD) table.
/// For a node with K states and P parent configuration combinations,
/// the CPD stores P(state | parent_config) as a flat array of shape (K, P).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cpd {
    /// Variable name
    pub variable: String,
    /// State names (e.g., ["0", "1"]) or feature labels
    pub states: Vec<String>,
    /// Parent variable names (order matters for parent config indexing)
    pub parents: Vec<String>,
    /// Parent state cardinalities, aligned with `parents`
    pub parent_card: Vec<usize>,
    /// CPD values: shape (states.len(), total_parent_configs)
    /// values[state_idx][parent_config_idx] = P(state | parent_config)
    pub values: Vec<Vec<f64>>,
}

impl Cpd {
    /// Create a new CPD with uniform distribution.
    pub fn new(variable: String, states: Vec<String>, parents: Vec<String>, parent_card: Vec<usize>) -> Self {
        let n_configs = parent_card.iter().product::<usize>().max(1);
        let n_states = states.len().max(1);
        let uniform = 1.0 / n_states as f64;
        let values = vec![vec![uniform; n_configs]; n_states];
        Self {
            variable,
            states,
            parents,
            parent_card,
            values,
        }
    }

    /// Get the index of a state value in the states list.
    pub fn state_index(&self, state: &str) -> usize {
        self.states.iter().position(|s| s == state).unwrap_or(0)
    }

    /// Get probability: P(variable=state | evidence), where evidence maps parent→state.
    pub fn probability(&self, state: &str, evidence: &HashMap<String, String>) -> f64 {
        let si = self.state_index(state);
        let pi = self.parent_config_index(evidence);
        if si < self.values.len() && pi < self.values[si].len() {
            self.values[si][pi]
        } else {
            0.0
        }
    }

    /// Compute the index into the parent configuration dimension.
    /// Parents are assumed to be ordered, each with cardinality `parent_card[i]`.
    /// Evidence maps parent variable name → string state.
    pub fn parent_config_index(&self, evidence: &HashMap<String, String>) -> usize {
        let mut idx = 0;
        let mut stride = 1;
        // Iterate in reverse to match the typical convention (last parent varies fastest)
        for pi in (0..self.parents.len()).rev() {
            let p_var = &self.parents[pi];
            let p_state = evidence.get(p_var).map(|s| s.as_str()).unwrap_or("0");
            let p_si = self
                .states_of_parent(pi)
                .iter()
                .position(|s| s == p_state)
                .unwrap_or(0);
            idx += p_si * stride;
            stride *= self.parent_card[pi].max(1);
        }
        idx
    }

    /// Generate list of valid state names for a given parent index.
    fn states_of_parent(&self, pi: usize) -> Vec<String> {
        // We generate states from 0..cardinality and convert to string.
        // For binary variables this is ["0", "1"].
        let card = self.parent_card.get(pi).copied().unwrap_or(2);
        (0..card).map(|i| i.to_string()).collect()
    }
}

/// A simple Bayesian Network (DAG + CPDs).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianNetwork {
    /// Directed edges: (parent, child)
    pub edges: Vec<(String, String)>,
    /// CPDs keyed by node name
    pub cpds: HashMap<String, Cpd>,
    /// Topological ordering of nodes
    pub node_order: Vec<String>,
    /// Target variable
    pub target: String,
    /// All node names
    pub nodes: Vec<String>,
}

impl BayesianNetwork {
    /// Create a new BN from edges.
    pub fn new(edges: Vec<(String, String)>) -> Self {
        let mut nodes_set: HashSet<String> = HashSet::new();
        for (p, c) in &edges {
            nodes_set.insert(p.clone());
            nodes_set.insert(c.clone());
        }
        let nodes: Vec<String> = nodes_set.into_iter().collect();
        Self {
            edges: edges.clone(),
            cpds: HashMap::new(),
            node_order: topological_sort(&nodes, &edges),
            target: "disturbance_flag".to_string(),
            nodes,
        }
    }

    /// Add a CPD for a node.
    pub fn add_cpd(&mut self, cpd: Cpd) {
        self.cpds.insert(cpd.variable.clone(), cpd);
    }

    /// Variable elimination: compute P(query_var = query_state | evidence).
    ///
    /// evidence: map of variable_name → state_string (e.g., {"IF_001": "1"})
    /// Returns probability of query_var = query_state.
    pub fn query(
        &self,
        query_var: &str,
        query_state: &str,
        evidence: &HashMap<String, String>,
    ) -> f64 {
        // Collect all relevant factors (CPDs)
        let _all_nodes: HashSet<String> = self.nodes.iter().cloned().collect();

        // Variables to eliminate: all non-query, non-evidence nodes
        let elim_order: Vec<String> = self
            .node_order
            .iter()
            .filter(|n| {
                n.as_str() != query_var && !evidence.contains_key(n.as_str())
            })
            .cloned()
            .collect();

        // Eliminate each variable
        let mut factors: Vec<ConditionalFactor> = self
            .cpds
            .values()
            .map(|cpd| ConditionalFactor::from_cpd(cpd, evidence))
            .collect();

        for var in &elim_order {
            factors = eliminate_variable(&factors, var);
        }

        // Multiply remaining factors and query
        let result = multiply_and_query(&factors, query_var, query_state, evidence);

        // Normalize
        if result.denom > 0.0 {
            result.num / result.denom
        } else {
            // Fallback: try just using the target CPD
            if let Some(cpd) = self.cpds.get(query_var) {
                let ev = evidence.clone();
                // For nodes with parents, use the target's parents
                if cpd.parents.is_empty() {
                    return cpd.probability(query_state, &ev);
                }
            }
            0.5
        }
    }

    /// Check model validity (CPDs sum to 1).
    pub fn check_model(&self) -> bool {
        for cpd in self.cpds.values() {
        let n_configs = cpd.parent_card.iter().product::<usize>().max(1);
            for pi in 0..n_configs {
                let sum: f64 = cpd.values.iter().map(|row| row.get(pi).copied().unwrap_or(0.0)).sum();
                if (sum - 1.0).abs() > 1e-6 {
                    return false;
                }
            }
        }
        true
    }
}

/// A factor used in variable elimination.
#[derive(Debug, Clone)]
struct ConditionalFactor {
    variables: Vec<String>,
    /// Flattened values table. For binary variables: [P(0,0), P(0,1), P(1,0), P(1,1)]
    values: Vec<f64>,
    // Cardinality of each variable
    cards: Vec<usize>,
}

impl ConditionalFactor {
    fn from_cpd(cpd: &Cpd, evidence: &HashMap<String, String>) -> Self {
        // Determine variables in this factor: node + parents
        let mut variables = cpd.parents.clone();
        variables.push(cpd.variable.clone());

        // Reduce evidence: if a variable is observed, fix it
        let fixed_vars: HashMap<String, String> = evidence
            .iter()
            .filter(|(k, _)| variables.contains(*k))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let n_states = cpd.states.len();
        let _n_configs = cpd.parent_card.iter().product::<usize>().max(1);

        // Build flattened table
        let mut cards: Vec<usize> = Vec::new();
        for v in &variables {
            if fixed_vars.contains_key(v) {
                cards.push(1); // fixed variable has 1 state
            } else if *v == cpd.variable {
                cards.push(n_states);
            } else {
                let pi = cpd.parents.iter().position(|p| p == v).unwrap_or(0);
                cards.push(cpd.parent_card.get(pi).copied().unwrap_or(2));
            }
        }

        let total_size: usize = cards.iter().product();
        let mut values = vec![0.0; total_size];

        // Iterate over all parent configs and node states, filling in
        // This is a simplified version for binary/moderate-size networks
        fill_factor_values(&mut values, &variables, &cards, &fixed_vars, cpd, 0, 0, &mut vec![0; 10]);

        Self {
            variables: variables.into_iter().filter(|v| !fixed_vars.contains_key(v)).collect(),
            values: values.into_iter().filter(|_| true).collect(),
            cards: cards.into_iter().filter(|&c| c > 1).collect(),
        }
    }
}

fn fill_factor_values(
    values: &mut [f64],
    variables: &[String],
    cards: &[usize],
    fixed: &HashMap<String, String>,
    cpd: &Cpd,
    depth: usize,
    base_idx: usize,
    state_assignment: &mut [usize],
) {
    if depth == variables.len() {
        // All variables assigned - compute probability
        let mut evidence = fixed.clone();
        let mut var_state: Option<String> = None;
        for (i, var) in variables.iter().enumerate() {
            if !fixed.contains_key(var) {
                evidence.insert(var.clone(), state_assignment[i].to_string());
                if var.as_str() == cpd.variable.as_str() {
                    var_state = Some(state_assignment[i].to_string());
                }
            }
        }
        let prob = cpd.probability(&var_state.unwrap_or_else(|| "0".to_string()), &evidence);
        values[base_idx] = prob;
        return;
    }

    let v = &variables[depth];
    if fixed.contains_key(v.as_str()) {
        fill_factor_values(values, variables, cards, fixed, cpd, depth + 1, base_idx, state_assignment);
    } else {
        let card = cards[depth];
        let stride: usize = cards[(depth + 1)..].iter().product();
        for si in 0..card {
            state_assignment[depth] = si;
            fill_factor_values(
                values,
                variables,
                cards,
                fixed,
                cpd,
                depth + 1,
                base_idx + si * stride,
                state_assignment,
            );
        }
    }
}

/// Eliminate a variable by summing over all its states.
fn eliminate_variable(factors: &[ConditionalFactor], var: &str) -> Vec<ConditionalFactor> {
    let mut relevant: Vec<ConditionalFactor> = Vec::new();
    let mut others: Vec<ConditionalFactor> = Vec::new();

    for f in factors {
        if f.variables.contains(&var.to_string()) {
            relevant.push(f.clone());
        } else {
            others.push(f.clone());
        }
    }

    if relevant.is_empty() {
        return others;
    }

    // Multiply relevant factors, then sum over var
    let mut combined = relevant[0].clone();
    for f in &relevant[1..] {
        combined = multiply_factors(&combined, f);
    }

    // Sum out var
    let combined = sum_out(&combined, var);
    others.push(combined);
    others
}

/// Multiply two factors.
fn multiply_factors(f1: &ConditionalFactor, f2: &ConditionalFactor) -> ConditionalFactor {
    let mut all_vars: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for v in &f1.variables {
        if !seen.contains(v) {
            all_vars.push(v.clone());
            seen.insert(v.clone());
        }
    }
    for v in &f2.variables {
        if !seen.contains(v) {
            all_vars.push(v.clone());
            seen.insert(v.clone());
        }
    }

    let mut cards: Vec<usize> = Vec::new();
    for v in &all_vars {
        let c1 = f1.variables.iter().position(|x| x == v).map(|i| f1.cards[i]).unwrap_or(1);
        let c2 = f2.variables.iter().position(|x| x == v).map(|i| f2.cards[i]).unwrap_or(1);
        cards.push(c1.max(c2));
    }

    let total: usize = cards.iter().product();
    let mut values = vec![0.0; total];

    // Index assignment helper
    let mut f1_idx_offsets = vec![0; all_vars.len()];
    let mut f2_idx_offsets = vec![0; all_vars.len()];
    for (ai, av) in all_vars.iter().enumerate() {
        if let Some(pi) = f1.variables.iter().position(|x| x == av) {
            f1_idx_offsets[ai] = pi;
        } else {
            f1_idx_offsets[ai] = usize::MAX;
        }
        if let Some(pi) = f2.variables.iter().position(|x| x == av) {
            f2_idx_offsets[ai] = pi;
        } else {
            f2_idx_offsets[ai] = usize::MAX;
        }
    }

    for ti in 0..total {
        let mut idx1 = 0;
        let mut idx2 = 0;
        let mut temp = ti;
        for (ai, card) in cards.iter().enumerate().rev() {
            let si = temp % card;
            temp /= card;
            let o1 = f1_idx_offsets[ai];
            let o2 = f2_idx_offsets[ai];
            if o1 != usize::MAX {
                let stride1: usize = f1.cards[(o1 + 1)..].iter().product();
                idx1 += si * stride1;
            }
            if o2 != usize::MAX {
                let stride2: usize = f2.cards[(o2 + 1)..].iter().product();
                idx2 += si * stride2;
            }
        }
        values[ti] = f1.values.get(idx1).copied().unwrap_or(0.0)
            * f2.values.get(idx2).copied().unwrap_or(0.0);
    }

    ConditionalFactor {
        variables: all_vars,
        values,
        cards,
    }
}

/// Sum out a variable from a factor.
fn sum_out(f: &ConditionalFactor, var: &str) -> ConditionalFactor {
    let var_idx = f.variables.iter().position(|v| v == var);
    if var_idx.is_none() {
        return f.clone();
    }
    let var_idx = var_idx.unwrap();
    let var_card = f.cards[var_idx];

    let new_vars: Vec<String> = f.variables.iter().enumerate()
        .filter(|(i, _)| *i != var_idx)
        .map(|(_, v)| v.clone())
        .collect();

    let new_cards: Vec<usize> = f.cards.iter().enumerate()
        .filter(|(i, _)| *i != var_idx)
        .map(|(_, &c)| c)
        .collect();

    let new_total: usize = new_cards.iter().product();
    let mut values = vec![0.0; new_total.max(1)];

    if new_total == 0 {
        // Only variable was the one being summed out
        let sum: f64 = f.values.iter().sum();
        return ConditionalFactor {
            variables: Vec::new(),
            values: vec![sum],
            cards: Vec::new(),
        };
    }

    // Sum over var_card states for each combination of remaining variables
    let stride_before: usize = f.cards[(var_idx + 1)..].iter().product();
    let var_stride = stride_before;
    let block_stride = var_stride * var_card;

    for old_idx in 0..f.values.len() {
        let new_idx = (old_idx / block_stride) * var_stride + (old_idx % var_stride);
        values[new_idx] += f.values[old_idx];
    }

    ConditionalFactor {
        variables: new_vars,
        values,
        cards: new_cards,
    }
}

#[derive(Debug)]
struct QueryResult {
    num: f64,
    denom: f64,
}

/// Multiply all factors and evaluate query.
fn multiply_and_query(
    factors: &[ConditionalFactor],
    query_var: &str,
    query_state: &str,
    _evidence: &HashMap<String, String>,
) -> QueryResult {
    if factors.is_empty() {
        return QueryResult { num: 0.5, denom: 1.0 };
    }

    // Combine all factors
    let mut combined = factors[0].clone();
    for f in &factors[1..] {
        combined = multiply_factors(&combined, f);
    }

    // If only query_var remains, evaluate
    let var_idx = combined.variables.iter().position(|v| v == query_var);
    match var_idx {
        Some(vi) => {
            let var_card = combined.cards[vi];
            let stride = combined.cards[(vi + 1)..].iter().product::<usize>();
            let block = stride * var_card;

            let mut probs = Vec::new();
            if combined.values.is_empty() {
                return QueryResult { num: 0.5, denom: 1.0 };
            }
            for si in 0..var_card {
                let mut sum = 0.0;
                for bi in 0..(combined.values.len() / block).max(1) {
                    let idx = bi * block + si * stride;
                    if idx < combined.values.len() {
                        sum += combined.values[idx];
                    }
                }
                probs.push(sum);
            }

            let total: f64 = probs.iter().sum();
            let si = query_state.parse::<usize>().unwrap_or(0);
            let num = probs.get(si).copied().unwrap_or(0.0);
            QueryResult { num, denom: total }
        }
        None => {
            // Query var not in remaining factors — uniform
            QueryResult { num: 0.5, denom: 1.0 }
        }
    }
}

// ── Topological sort ───────────────────────────────────────────

fn topological_sort(nodes: &[String], edges: &[(String, String)]) -> Vec<String> {
    let mut in_degree: HashMap<String, usize> = nodes.iter().map(|n| (n.clone(), 0)).collect();
    let mut adj: HashMap<String, Vec<String>> = nodes.iter().map(|n| (n.clone(), Vec::new())).collect();

    for (p, c) in edges {
        *in_degree.entry(c.clone()).or_insert(0) += 1;
        adj.entry(p.clone()).or_default().push(c.clone());
    }

    let mut queue: Vec<String> = in_degree
        .iter()
        .filter(|(_, &d)| d == 0)
        .map(|(n, _)| n.clone())
        .collect();

    let mut sorted = Vec::new();
    while let Some(node) = queue.pop() {
        sorted.push(node.clone());
        if let Some(children) = adj.get(&node) {
            for child in children {
                if let Some(d) = in_degree.get_mut(child) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push(child.clone());
                    }
                }
            }
        }
    }

    // Add any remaining nodes (cycles)
    for node in nodes {
        if !sorted.contains(node) {
            sorted.push(node.clone());
        }
    }

    sorted
}

// ── Fallback model ─────────────────────────────────────────────

/// Fallback attribution model using logistic regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackAttributionModelData {
    pub edges: Vec<(String, String)>,
    pub factor_ids: Vec<String>,
    pub target: String,
    pub weights: Vec<f64>,
    pub intercept: f64,
    pub base_rate: f64,
    pub backend: String,
    pub reason: String,
}

// ── BayesianAttributionEngine ──────────────────────────────────

/// Constrained Bayesian Attribution Engine.
///
/// Builds a fixed-structure Bayesian network from selected inducing factors
/// and feature variables. Falls back to logistic regression if BN inference
/// fails or is unsuitable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BayesianAttributionEngine {
    pub target: String,
    pub random_state: u64,
    pub dominant_threshold: f64,
    pub synergy_threshold: f64,
    /// The trained Bayesian network
    pub model: Option<BayesianNetwork>,
    /// Backend used: "bayesian_network" or "logistic_fallback"
    pub model_backend: String,
    /// Network features (feature columns used in the BN)
    pub network_features: Vec<String>,
    /// Factor columns (IF_001, IF_002, etc.)
    pub factor_columns: Vec<String>,
    /// Edge list
    pub edges: Vec<(String, String)>,
    /// Fallback model data
    pub fallback: Option<FallbackAttributionModelData>,
    /// Reason for using fallback
    pub fallback_reason: String,
}

impl BayesianAttributionEngine {
    pub fn new(
        target: String,
        random_state: u64,
        dominant_threshold: f64,
        synergy_threshold: f64,
    ) -> Self {
        Self {
            target,
            random_state,
            dominant_threshold,
            synergy_threshold,
            model: None,
            model_backend: "none".to_string(),
            network_features: Vec::new(),
            factor_columns: Vec::new(),
            edges: Vec::new(),
            fallback: None,
            fallback_reason: String::new(),
        }
    }

    /// Build a fixed edge structure from factors and network features.
    pub fn build_fixed_edges(
        selected_factors: &[InducingFactor],
    ) -> (Vec<(String, String)>, Vec<String>, Vec<String>) {
        let mut network_features: HashSet<String> = HashSet::new();
        for factor in selected_factors {
            for item in &factor.items {
                if let Some((feature, _state)) = item.split_once('=') {
                    network_features.insert(feature.to_string());
                }
            }
        }
        let network_features_sorted: Vec<String> = {
            let mut v: Vec<String> = network_features.into_iter().collect();
            v.sort();
            v
        };

        let factor_ids: Vec<String> = selected_factors
            .iter()
            .map(|f| f.factor_id.clone())
            .collect();

        let mut edges = Vec::new();
        for factor in selected_factors {
            for item in &factor.items {
                if let Some((feature, _state)) = item.split_once('=') {
                    edges.push((feature.to_string(), factor.factor_id.clone()));
                }
            }
            // Default target name
            edges.push((factor.factor_id.clone(), "disturbance_flag".to_string()));
        }

        // Deduplicate while preserving order
        let mut seen = HashSet::new();
        let unique_edges: Vec<(String, String)> = edges
            .into_iter()
            .filter(|e| seen.insert(e.clone()))
            .collect();

        (unique_edges, network_features_sorted, factor_ids)
    }

    /// Estimate CPDs using maximum likelihood with Laplace (+1) smoothing.
    pub fn estimate_cpds(
        bn_df: &[HashMap<String, String>],
        edges: &[(String, String)],
        nodes: &[String],
        factor_ids: &[String],
        network_features: &[String],
        target: &str,
    ) -> HashMap<String, Cpd> {
        let _n_samples = bn_df.len();
        let mut cpds = HashMap::new();

        // Build parent map
        let mut parents: HashMap<String, Vec<String>> = HashMap::new();
        for (p, c) in edges {
            parents.entry(c.clone()).or_default().push(p.clone());
        }

        for node in nodes {
            let node_parents = parents.get(node).cloned().unwrap_or_default();
            let is_target = node.as_str() == target;

            // Determine state cardinality
            let node_card: usize = if is_target || factor_ids.contains(node) {
                2 // binary (0/1)
            } else if network_features.contains(node) {
                // Count unique states in data
                let unique: HashSet<String> = bn_df
                    .iter()
                    .filter_map(|row| row.get(node))
                    .cloned()
                    .collect();
                unique.len().max(2) // at least 2 for features
            } else {
                2
            };

            let node_states: Vec<String> = if is_target || factor_ids.contains(node) {
                vec!["0".to_string(), "1".to_string()]
            } else {
                // For features: collect unique states from data, sorted
                let mut unique: Vec<String> = bn_df
                    .iter()
                    .filter_map(|row| row.get(node))
                    .cloned()
                    .collect();
                unique.sort();
                unique.dedup();
                if unique.len() < 2 {
                    vec!["0".to_string(), "1".to_string()]
                } else {
                    unique
                }
            };

            let parent_cards: Vec<usize> = node_parents
                .iter()
                .map(|p| {
                    if factor_ids.contains(p) || p == target {
                        2
                    } else {
                        let unique: HashSet<String> = bn_df
                            .iter()
                            .filter_map(|row| row.get(p))
                            .cloned()
                            .collect();
                        unique.len().max(2)
                    }
                })
                .collect();

            let n_configs = parent_cards.iter().product::<usize>().max(1);

            // Count occurrences: counts[state_idx][config_idx]
            let mut counts = vec![vec![1.0_f64; n_configs]; node_card]; // Laplace +1

            for row in bn_df {
                let state_str = row.get(node).map(|s| s.as_str()).unwrap_or("0");
                let si = node_states.iter().position(|s| s == state_str).unwrap_or(0);

                let mut pi = 0;
                let mut stride = 1;
                for (ppi, parent) in node_parents.iter().enumerate().rev() {
                    let p_state = row.get(parent).map(|s| s.as_str()).unwrap_or("0");
                    let card = parent_cards[ppi];
                    // Simple index: use state as number or hash
                    let p_si = p_state.parse::<usize>().unwrap_or(0) % card;
                    pi += p_si * stride;
                    stride *= card;
                }
                pi = pi.min(n_configs - 1);

                counts[si][pi] += 1.0;
            }

            // Normalize to probabilities
            let mut values = vec![vec![0.0; n_configs]; node_card];
            for ci in 0..n_configs {
                let sum: f64 = counts.iter().map(|row| row[ci]).sum();
                for si in 0..node_card {
                    values[si][ci] = if sum > 0.0 { counts[si][ci] / sum } else { 1.0 / node_card as f64 };
                }
            }

            cpds.insert(
                node.clone(),
                Cpd {
                    variable: node.clone(),
                    states: node_states,
                    parents: node_parents,
                    parent_card: parent_cards,
                    values,
                },
            );
        }

        cpds
    }

    /// Fit the Bayesian network on labeled data.
    ///
    /// `label_df`: discretized label data (HashMap per row)
    /// `training_targets`: disturbance_flag values (0.0 or 1.0)
    /// `selected_factors`: inducing factors
    /// `feature_columns`: all feature column names (for reference)
    pub fn fit(
        &mut self,
        label_df: &[HashMap<String, String>],
        training_targets: &[f64],
        selected_factors: &[InducingFactor],
    ) {
        let (edges, network_features, factor_columns) =
            Self::build_fixed_edges(selected_factors);

        self.edges = edges.clone();
        self.network_features = network_features.clone();
        self.factor_columns = factor_columns.clone();

        // Build BN dataframe
        let mut bn_df: Vec<HashMap<String, String>> = Vec::with_capacity(label_df.len());
        for (i, row) in label_df.iter().enumerate() {
            let mut bn_row = HashMap::new();
            // Copy network feature states
            for feat in &self.network_features {
                if let Some(val) = row.get(feat) {
                    bn_row.insert(feat.clone(), val.clone());
                } else {
                    bn_row.insert(feat.clone(), "missing".to_string());
                }
            }
            // Add factor activations
            for factor in selected_factors {
                let active = factor_activation_for_row_int(row, &[factor.clone()])[0];
                bn_row.insert(factor.factor_id.clone(), active.to_string());
            }
            // Add target
            let target_val = if i < training_targets.len() {
                training_targets[i] as i32
            } else {
                0
            };
            bn_row.insert(self.target.clone(), target_val.to_string());
            bn_df.push(bn_row);
        }

        // Collect all nodes
        let mut all_nodes: Vec<String> = self.network_features.clone();
        for fid in &self.factor_columns {
            all_nodes.push(fid.clone());
        }
        all_nodes.push(self.target.clone());

        // Try Bayesian network
        let mut bn = BayesianNetwork::new(edges);
        let cpds = Self::estimate_cpds(
            &bn_df,
            &bn.edges,
            &all_nodes,
            &self.factor_columns,
            &self.network_features,
            &self.target,
        );
        bn.cpds = cpds;

        // Check if BN is valid
        if bn.check_model() && !bn.cpds.is_empty() {
            self.model = Some(bn);
            self.model_backend = "bayesian_network".to_string();
            self.fallback = None;
            self.fallback_reason = String::new();
        } else {
            // Fall back to logistic regression
            self.model = None;
            self.model_backend = "logistic_fallback".to_string();
            self.fallback_reason = "BN check failed or empty CPDs".to_string();
            self.build_fallback(label_df, training_targets, selected_factors);
        }
    }

    /// Build logistic regression fallback.
    fn build_fallback(
        &mut self,
        label_df: &[HashMap<String, String>],
        training_targets: &[f64],
        selected_factors: &[InducingFactor],
    ) {
        let X = build_factor_activation(label_df, selected_factors);

        let n_pos: f64 = training_targets.iter().filter(|&&v| v > 0.5).count() as f64;
        let n_neg = training_targets.len() as f64 - n_pos;
        let base_rate = if training_targets.is_empty() {
            0.0
        } else {
            n_pos / (training_targets.len() as f64).max(1.0)
        };

        let pos_weight = training_targets.len() as f64 / (2.0 * n_pos.max(1.0));
        let neg_weight = training_targets.len() as f64 / (2.0 * n_neg.max(1.0));
        let class_weight = vec![neg_weight, pos_weight];

        let (weights, intercept) = fit_logistic_binary_internal(
            &X,
            training_targets,
            Some(&class_weight),
            100,
            1.0,
        );

        let factor_ids: Vec<String> = selected_factors.iter().map(|f| f.factor_id.clone()).collect();

        self.fallback = Some(FallbackAttributionModelData {
            edges: self.edges.clone(),
            factor_ids,
            target: self.target.clone(),
            weights,
            intercept,
            base_rate,
            backend: "logistic_fallback".to_string(),
            reason: self.fallback_reason.clone(),
        });
    }

    /// Compute inference result for a single sample.
    /// Returns probability of disturbance (target=1) given evidence.
    pub fn infer_single(
        &self,
        row: &HashMap<String, String>,
        factors: &[InducingFactor],
    ) -> f64 {
        if let Some(ref bn) = self.model {
            // Build evidence from factor activations
            let mut evidence = HashMap::new();
            // Add network feature evidence
            for feat in &self.network_features {
                if let Some(val) = row.get(feat) {
                    evidence.insert(feat.clone(), val.clone());
                }
            }
            // Add factor evidence
            for factor in factors {
                let active = factor_activation_for_row_int(row, &[factor.clone()])[0];
                evidence.insert(factor.factor_id.clone(), active.to_string());
            }

            bn.query(&self.target, "1", &evidence)
        } else if let Some(ref fb) = self.fallback {
            // Logistic regression fallback
            let activation = factor_activation_for_row_int(row, factors);
            let mut lin = fb.intercept;
            for (j, &val) in activation.iter().enumerate() {
                if j < fb.weights.len() {
                    lin += fb.weights[j] * val as f64;
                }
            }
            // Sigmoid
            let clamped = lin.clamp(-30.0, 30.0);
            1.0 / (1.0 + (-clamped).exp())
        } else {
            0.5
        }
    }

    /// Attribute a single sample — determine which factors contribute to disturbance.
    /// Returns: (distribution_type, top_factor, top_prob, bayesian_prob)
    pub fn attribute_sample(
        &self,
        row: &HashMap<String, String>,
        factors: &[InducingFactor],
    ) -> AttributionResult {
        let bayesian_prob = self.infer_single(row, factors);

        // Compute per-factor contributions using factor activations
        let mut factor_probs: Vec<(String, f64)> = Vec::new();
        for factor in factors {
            let active = factor_activation_for_row_int(row, &[factor.clone()])[0];
            let prob = if active > 0 {
                // Factor active: use factor's relative risk to estimate contribution
                (factor.relative_risk / (factor.relative_risk + 1.0)).clamp(0.0, 0.99)
            } else {
                0.0
            };
            factor_probs.push((factor.factor_id.clone(), prob));
        }

        // Sort by probability descending
        factor_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let (attribution_type, top_factor, top_prob) = if factor_probs.is_empty() {
            ("background_or_unknown".to_string(), "none".to_string(), 0.0)
        } else {
            let top = &factor_probs[0];
            let second = factor_probs.get(1);
            if top.1 >= self.dominant_threshold
                && second.map_or(true, |s| top.1 - s.1 >= 0.05)
            {
                ("single_factor_dominant".to_string(), top.0.clone(), top.1)
            } else if top.1 + second.map(|s| s.1).unwrap_or(0.0) >= self.synergy_threshold {
                ("multi_factor_synergy".to_string(), top.0.clone(), top.1)
            } else {
                ("background_or_unknown".to_string(), top.0.clone(), top.1)
            }
        };

        AttributionResult {
            attribution_type,
            top_factor,
            top_probability: top_prob,
            bayesian_probability: bayesian_prob,
            backend: self.model_backend.clone(),
        }
    }
}

/// Result of attributing a single sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionResult {
    pub attribution_type: String,
    pub top_factor: String,
    pub top_probability: f64,
    pub bayesian_probability: f64,
    pub backend: String,
}

use crate::model::fit_logistic_binary_internal;

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_row(vals: &[(&str, &str)]) -> HashMap<String, String> {
        let mut row = HashMap::new();
        for (k, v) in vals {
            row.insert(k.to_string(), v.to_string());
        }
        row
    }

    fn make_test_factor(id: &str) -> InducingFactor {
        InducingFactor {
            factor_id: id.to_string(),
            name: format!("{}_name", id),
            semantics: format!("{}_sem", id),
            items: vec![format!("feat_a=high")],
            support_event: 0.5, support_background: 0.1, relative_risk: 5.0, stability: 1.0,
            score: 1.0, count_event: 50, count_background: 10, n_event: 100, n_background: 100,
            rr_ci_lower: 2.0, rr_ci_upper: 10.0, p_value: 0.01, fdr_significant: true,
            likelihood_ratio: 5.0, true_relative_risk: 5.0,
        }
    }

    #[test]
    fn test_topological_sort() {
        let nodes = vec!["a".to_string(), "b".to_string(), "c".to_string(), "d".to_string()];
        let edges = vec![
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "c".to_string()),
            ("a".to_string(), "d".to_string()),
        ];
        let sorted = topological_sort(&nodes, &edges);
        // a must come before b, b before c
        let pos_a = sorted.iter().position(|n| n == "a").unwrap();
        let pos_b = sorted.iter().position(|n| n == "b").unwrap();
        let pos_c = sorted.iter().position(|n| n == "c").unwrap();
        assert!(pos_a < pos_b, "a before b");
        assert!(pos_b < pos_c, "b before c");
    }

    #[test]
    fn test_build_fixed_edges() {
        let f1 = InducingFactor {
            factor_id: "IF_001".to_string(),
            name: "f1".to_string(),
            semantics: "f1".to_string(),
            items: vec!["a=high".to_string(), "b=near".to_string()],
            support_event: 0.5, support_background: 0.1, relative_risk: 5.0, stability: 1.0,
            score: 1.0, count_event: 50, count_background: 10, n_event: 100, n_background: 100,
            rr_ci_lower: 2.0, rr_ci_upper: 10.0, p_value: 0.01, fdr_significant: true,
            likelihood_ratio: 5.0, true_relative_risk: 5.0,
        };

        let (edges, network_features, factor_columns) =
            BayesianAttributionEngine::build_fixed_edges(&[f1]);

        assert!(edges.contains(&("a".to_string(), "IF_001".to_string())));
        assert!(edges.contains(&("b".to_string(), "IF_001".to_string())));
        assert!(edges.contains(&("IF_001".to_string(), "disturbance_flag".to_string())));
        assert!(network_features.contains(&"a".to_string()));
        assert!(network_features.contains(&"b".to_string()));
        assert_eq!(factor_columns, vec!["IF_001"]);
    }

    #[test]
    fn test_cpd_estimation_binary() {
        let bn_df = vec![
            make_row(&[("a", "high"), ("IF_001", "1"), ("disturbance_flag", "1")]),
            make_row(&[("a", "high"), ("IF_001", "1"), ("disturbance_flag", "1")]),
            make_row(&[("a", "low"), ("IF_001", "0"), ("disturbance_flag", "0")]),
            make_row(&[("a", "low"), ("IF_001", "0"), ("disturbance_flag", "0")]),
        ];

        let cpds = BayesianAttributionEngine::estimate_cpds(
            &bn_df,
            &[("a".to_string(), "IF_001".to_string()), ("IF_001".to_string(), "disturbance_flag".to_string())],
            &["a".to_string(), "IF_001".to_string(), "disturbance_flag".to_string()],
            &["IF_001".to_string()],
            &["a".to_string()],
            "disturbance_flag",
        );

        assert!(cpds.contains_key("IF_001"));
        assert!(cpds.contains_key("disturbance_flag"));
        assert!(cpds.contains_key("a"));
    }

    #[test]
    fn test_bn_query_simple() {
        // Build a simple BN: a → IF → target
        let edges = vec![
            ("a".to_string(), "IF_001".to_string()),
            ("IF_001".to_string(), "disturbance_flag".to_string()),
        ];

        let mut bn = BayesianNetwork::new(edges);

        // a has 2 states: "high", "low"
        let mut cpds = HashMap::new();

        // P(a)
        cpds.insert("a".to_string(), Cpd {
            variable: "a".to_string(),
            states: vec!["high".to_string(), "low".to_string()],
            parents: vec![],
            parent_card: vec![],
            values: vec![vec![0.8], vec![0.2]], // P(a=high)=0.8, P(a=low)=0.2
        });

        // P(IF_001 | a): IF_001 has states ["0", "1"]
        // IF_001=1 | a=high = 0.9, IF_001=1 | a=low = 0.1
        cpds.insert("IF_001".to_string(), Cpd {
            variable: "IF_001".to_string(),
            states: vec!["0".to_string(), "1".to_string()],
            parents: vec!["a".to_string()],
            parent_card: vec![2],
            values: vec![
                vec![0.1, 0.9],  // P(IF_001=0 | a=high)=0.1, P(IF_001=0 | a=low)=0.9
                vec![0.9, 0.1],  // P(IF_001=1 | ...)
            ],
        });

        // P(target | IF_001)
        cpds.insert("disturbance_flag".to_string(), Cpd {
            variable: "disturbance_flag".to_string(),
            states: vec!["0".to_string(), "1".to_string()],
            parents: vec!["IF_001".to_string()],
            parent_card: vec![2],
            values: vec![
                vec![0.95, 0.3], // P(target=0 | IF=0)=0.95, P(target=0 | IF=1)=0.3
                vec![0.05, 0.7], // P(target=1 | ...)
            ],
        });

        bn.cpds = cpds;

        // Query: P(target=1 | a=high)
        let mut evidence = HashMap::new();
        evidence.insert("a".to_string(), "high".to_string());
        let prob = bn.query("disturbance_flag", "1", &evidence);

        // Expected: P(target=1|a=high) = P(target=1|IF=1)*P(IF=1|a=high) + P(target=1|IF=0)*P(IF=0|a=high)
        // = 0.7 * 0.9 + 0.05 * 0.1 = 0.63 + 0.005 = 0.635
        assert!((prob - 0.635).abs() < 0.05, "Expected ~0.635, got {}", prob);
    }

    #[test]
    fn test_bayesian_engine_structure() {
        let f1 = make_test_factor("IF_001");

        let label_df = vec![
            make_row(&[("feat_a", "high")]),
            make_row(&[("feat_a", "low")]),
        ];
        let targets = vec![1.0, 0.0];

        let mut engine = BayesianAttributionEngine::new(
            "disturbance_flag".to_string(),
            42,
            0.55,
            0.75,
        );

        engine.fit(&label_df, &targets, &[f1]);

        // Should have edges and network features
        assert!(!engine.edges.is_empty());
        assert!(!engine.network_features.is_empty());
        assert!(!engine.factor_columns.is_empty());

        // Either BN or fallback
        assert!(
            engine.model_backend == "bayesian_network" || engine.model_backend == "logistic_fallback"
        );
    }

    #[test]
    fn test_bayesian_engine_inference() {
        let f1 = make_test_factor("IF_001");

        // Create strongly separable data
        let label_df = vec![
            make_row(&[("feat_a", "high")]),
            make_row(&[("feat_a", "high")]),
            make_row(&[("feat_a", "low")]),
            make_row(&[("feat_a", "low")]),
        ];
        let targets = vec![1.0, 1.0, 0.0, 0.0];

        let mut engine = BayesianAttributionEngine::new(
            "disturbance_flag".to_string(),
            42,
            0.55,
            0.75,
        );

        engine.fit(&label_df, &targets, &[f1.clone()]);

        let prob = engine.infer_single(&make_row(&[("feat_a", "high")]), &[f1]);
        assert!(prob >= 0.0 && prob <= 1.0, "Prob out of range: {}", prob);
    }

    #[test]
    fn test_attribute_sample() {
        let f1 = make_test_factor("IF_001");

        let label_df = vec![
            make_row(&[("feat_a", "high")]),
            make_row(&[("feat_a", "high")]),
            make_row(&[("feat_a", "low")]),
        ];
        let targets = vec![1.0, 1.0, 0.0];

        let mut engine = BayesianAttributionEngine::new(
            "disturbance_flag".to_string(),
            42,
            0.55,
            0.75,
        );

        engine.fit(&label_df, &targets, &[f1.clone()]);

        let result = engine.attribute_sample(&make_row(&[("feat_a", "high")]), &[f1]);
        assert!(result.bayesian_probability >= 0.0 && result.bayesian_probability <= 1.0);
        assert!(!result.attribution_type.is_empty());
        assert!(!result.backend.is_empty());
    }
}

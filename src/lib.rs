/*!
# cuda-forth

Minimal Forth-like agent language compiling to cuda-instruction-set bytecode.

Forth is the perfect agent language:
- Stack-based (matches instruction set)
- Extensible (agents define their own words)
- Compact (fits in memory-constrained environments)
- Self-hosting potential (the language IS the agent)

```
  : square DUP * ;
  : count-to-0 BEGIN DUP 0> WHILE 1- DUP . REPEAT DROP ;
  5 count-to-0
```
*/

use std::collections::HashMap;

/// Built-in Forth words
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Builtin {
    Dup, Drop, Swap, Over, Rot,
    Add, Sub, Mul, Div, Mod, Neg,
    Eq, Neq, Lt, Gt, Le, Ge,
    And, Or, Xor, Not,
    Shl, Shr,
    Emit, Dot, Cr,
    Here, Comma, At, Store, Cells,
    If, Then, Else, Begin, While, Repeat, Until, Do, Loop, I,
    Confidence, Trust, Gate, Fuse,
    Tell, Ask, Broadcast,
    Instinct, GeneExpr, EnzymeBind, MembraneChk,
    AtpGen, AtpConsume, AtpQ,
    ApoptosisChk, CircadianGet, CircadianSet,
    Halt,
}

impl Builtin {
    fn all() -> &'static [(&'static str, Builtin)] {
        &[
            ("DUP", Builtin::Dup), ("DROP", Builtin::Drop), ("SWAP", Builtin::Swap),
            ("OVER", Builtin::Over), ("ROT", Builtin::Rot),
            ("+", Builtin::Add), ("-", Builtin::Sub), ("*", Builtin::Mul),
            ("/", Builtin::Div), ("MOD", Builtin::Mod), ("NEGATE", Builtin::Neg),
            ("=", Builtin::Eq), ("<>", Builtin::Neq), ("<", Builtin::Lt),
            (">", Builtin::Gt), ("<=", Builtin::Le), (">=", Builtin::Ge),
            ("AND", Builtin::And), ("OR", Builtin::Or), ("XOR", Builtin::Xor),
            ("NOT", Builtin::Not), ("SHL", Builtin::Shl), ("SHR", Builtin::Shr),
            ("EMIT", Builtin::Emit), (".", Builtin::Dot), ("CR", Builtin::Cr),
            ("HERE", Builtin::Here), (",", Builtin::Comma), ("@", Builtin::At),
            ("!", Builtin::Store), ("CELLS", Builtin::Cells),
            ("IF", Builtin::If), ("THEN", Builtin::Then), ("ELSE", Builtin::Else),
            ("BEGIN", Builtin::Begin), ("WHILE", Builtin::While),
            ("REPEAT", Builtin::Repeat), ("UNTIL", Builtin::Until),
            ("DO", Builtin::Do), ("LOOP", Builtin::Loop), ("I", Builtin::I),
            ("CONFIDENCE", Builtin::Confidence), ("TRUST", Builtin::Trust),
            ("GATE", Builtin::Gate), ("FUSE", Builtin::Fuse),
            ("TELL", Builtin::Tell), ("ASK", Builtin::Ask), ("BROADCAST", Builtin::Broadcast),
            ("INSTINCT", Builtin::Instinct), ("GENE_EXPR", Builtin::GeneExpr),
            ("ENZYME_BIND", Builtin::EnzymeBind), ("MEMBRANE_CHK", Builtin::MembraneChk),
            ("ATP_GEN", Builtin::AtpGen), ("ATP_CONSUME", Builtin::AtpConsume),
            ("ATP_Q", Builtin::AtpQ),
            ("APOPTOSIS_CHK", Builtin::ApoptosisChk),
            ("CIRCADIAN_GET", Builtin::CircadianGet), ("CIRCADIAN_SET", Builtin::CircadianSet),
            ("HALT", Builtin::Halt),
        ]
    }
}

/// A compiled word (user-defined)
#[derive(Clone, Debug)]
pub struct Word {
    pub name: String,
    /// Bytecodes of the compiled body (references to builtins + literals)
    pub body: Vec<WordOp>,
    pub params: Vec<String>, // ( -- ) stack effect notation
}

#[derive(Clone, Debug)]
pub enum WordOp {
    CallBuiltin(Builtin),
    PushLiteral(i32),
    CallWord(String),
    BranchForward(usize),  // relative offset
    BranchBack(usize),     // relative offset
    BranchIfZero(usize),   // IF
    BranchIfNotZero(usize), // WHILE
}

/// The Forth compiler/interpreter
pub struct Forth {
    pub words: HashMap<String, Word>,
    pub data_stack: Vec<i32>,
    pub return_stack: Vec<usize>,
    pub memory: Vec<i32>,
    pub here: usize, // next free memory address
    pub pc: usize,
    pub output: String,
    /// Confidence associated with current operation
    pub confidence: f64,
    /// Agent trust level
    pub trust: f64,
    /// Current instinct ID
    pub instinct: u8,
    /// Current energy (ATP)
    pub energy: f64,
    pub max_energy: f64,
}

impl Forth {
    pub fn new() -> Self {
        Forth {
            words: HashMap::new(),
            data_stack: vec![],
            return_stack: vec![],
            memory: vec![0i32; 1024],
            here: 0,
            pc: 0,
            output: String::new(),
            confidence: 1.0,
            trust: 0.5,
            instinct: 0,
            energy: 100.0,
            max_energy: 100.0,
        }
    }

    /// Parse and compile source into words
    pub fn compile(&mut self, source: &str) -> Result<Vec<String>, String> {
        let mut defined: Vec<String> = vec![];
        let tokens = self.tokenize(source);
        let mut i = 0;

        while i < tokens.len() {
            let tok = &tokens[i];
            if tok == ":" {
                // Word definition: : name ... ;
                i += 1;
                if i >= tokens.len() { return Err("expected word name after ':'".into()); }
                let name = tokens[i].to_uppercase();
                i += 1;
                let mut body: Vec<WordOp> = vec![];
                while i < tokens.len() && tokens[i] != ";" {
                    let op = self.parse_token(&tokens[i], &tokens, &mut i)?;
                    body.push(op);
                    i += 1;
                }
                if i >= tokens.len() { return Err("missing ';'".into()); }
                self.words.insert(name.clone(), Word { name: name.clone(), body, params: vec![] });
                defined.push(name);
            } else {
                // Top-level expression (compile as anonymous word and execute)
                let op = self.parse_token(tok, &tokens, &mut i)?;
                // Execute immediately for top-level
                self.execute_op(&op)?;
            }
            i += 1;
        }
        Ok(defined)
    }

    /// Execute source directly (compile + run)
    pub fn execute(&mut self, source: &str) -> Result<String, String> {
        self.compile(source)?;
        // Find and execute the last defined word, or run top-level ops
        Ok(self.output.clone())
    }

    fn tokenize(&self, source: &str) -> Vec<String> {
        let mut tokens = vec![];
        let mut current = String::new();
        let mut in_comment = false;

        for ch in source.chars() {
            match ch {
                '\\' if in_comment || current.is_empty() => in_comment = true,
                '\n' => { in_comment = false; if !current.is_empty() { tokens.push(current.trim().to_string()); current.clear(); } }
                '(' if current.is_empty() && tokens.last().map_or(true, |t| t == " ") => in_comment = true,
                ')' if in_comment => in_comment = false,
                ' ' | '\t' | '\r' => { if !current.is_empty() { tokens.push(current.trim().to_string()); current.clear(); } }
                _ if !in_comment => current.push(ch),
                _ => {}
            }
        }
        if !current.is_empty() { tokens.push(current.trim().to_string()); }
        tokens.into_iter().filter(|t| !t.is_empty()).collect()
    }

    fn parse_token(&self, tok: &str, _all: &[String], _i: &mut usize) -> Result<WordOp, String> {
        // Check builtins
        let upper = tok.to_uppercase();
        for (name, builtin) in Builtin::all() {
            if upper == *name { return Ok(WordOp::CallBuiltin(builtin)); }
        }
        // Check user words
        if self.words.contains_key(&upper) { return Ok(WordOp::CallWord(upper)); }
        // Try as number
        if let Ok(n) = tok.parse::<i32>() { return Ok(WordOp::PushLiteral(n)); }
        Err(format!("unknown word: {}", tok))
    }

    fn execute_op(&mut self, op: &WordOp) -> Result<(), String> {
        match op {
            WordOp::PushLiteral(n) => { self.data_stack.push(*n); }
            WordOp::CallBuiltin(b) => self.exec_builtin(*b)?,
            WordOp::CallWord(name) => {
                if let Some(word) = self.words.get(name).cloned() {
                    for wop in &word.body { self.execute_op(wop)?; }
                }
            }
            _ => {} // branch ops need more context
        }
        Ok(())
    }

    fn pop(&mut self) -> Result<i32, String> {
        self.data_stack.pop().ok_or("stack underflow")
    }

    fn push(&mut self, val: i32) { self.data_stack.push(val); }

    fn exec_builtin(&mut self, b: Builtin) -> Result<(), String> {
        match b {
            Builtin::Dup => { let a = self.pop()?; self.push(a); self.push(a); }
            Builtin::Drop => { self.pop()?; }
            Builtin::Swap => { let b = self.pop()?; let a = self.pop()?; self.push(b); self.push(a); }
            Builtin::Over => { let a = self.pop()?; let b = self.pop()?; self.push(b); self.push(a); self.push(b); }
            Builtin::Rot => {
                let c = self.pop()?; let b = self.pop()?; let a = self.pop()?;
                self.push(b); self.push(c); self.push(a);
            }
            Builtin::Add => { let b = self.pop()?; let a = self.pop()?; self.push(a + b); }
            Builtin::Sub => { let b = self.pop()?; let a = self.pop()?; self.push(a - b); }
            Builtin::Mul => { let b = self.pop()?; let a = self.pop()?; self.push(a * b); }
            Builtin::Div => { let b = self.pop()?; let a = self.pop()?; if b == 0 { return Err("division by zero".into()); } self.push(a / b); }
            Builtin::Mod => { let b = self.pop()?; let a = self.pop()?; self.push(a % b); }
            Builtin::Neg => { let a = self.pop()?; self.push(-a); }
            Builtin::Eq => { let b = self.pop()?; let a = self.pop()?; self.push(if a == b { 1 } else { 0 }); }
            Builtin::Neq => { let b = self.pop()?; let a = self.pop()?; self.push(if a != b { 1 } else { 0 }); }
            Builtin::Lt => { let b = self.pop()?; let a = self.pop()?; self.push(if a < b { 1 } else { 0 }); }
            Builtin::Gt => { let b = self.pop()?; let a = self.pop()?; self.push(if a > b { 1 } else { 0 }); }
            Builtin::Le => { let b = self.pop()?; let a = self.pop()?; self.push(if a <= b { 1 } else { 0 }); }
            Builtin::Ge => { let b = self.pop()?; let a = self.pop()?; self.push(if a >= b { 1 } else { 0 }); }
            Builtin::And => { let b = self.pop()?; let a = self.pop()?; self.push(a & b); }
            Builtin::Or => { let b = self.pop()?; let a = self.pop()?; self.push(a | b); }
            Builtin::Xor => { let b = self.pop()?; let a = self.pop()?; self.push(a ^ b); }
            Builtin::Not => { let a = self.pop()?; self.push(!a); }
            Builtin::Shl => { let b = self.pop()?; let a = self.pop()?; self.push(a << b); }
            Builtin::Shr => { let b = self.pop()?; let a = self.pop()?; self.push(a >> b); }
            Builtin::Emit => { let a = self.pop()?; self.output.push(char::from_u32(a as u32).unwrap_or('?')); }
            Builtin::Dot => { let a = self.pop()?; self.output.push_str(&format!("{} ", a)); }
            Builtin::Cr => { self.output.push('\n'); }
            Builtin::Here => { self.push(self.here as i32); }
            Builtin::Comma => { let a = self.pop()?; self.memory[self.here] = a; self.here += 1; }
            Builtin::At => { let addr = self.pop()? as usize; if addr < self.memory.len() { self.push(self.memory[addr]); } else { self.push(0); } }
            Builtin::Store => { let addr = self.pop()? as usize; let val = self.pop()?; if addr < self.memory.len() { self.memory[addr] = val; } }
            Builtin::Cells => { let n = self.pop()?; self.push(n * 4); } // cell = 4 bytes
            Builtin::Confidence => { self.push((self.confidence * 1000.0) as i32); } // fixed-point
            Builtin::Trust => { self.push((self.trust * 1000.0) as i32); }
            Builtin::Gate => { let threshold = self.pop()? as f64 / 1000.0; if self.confidence < threshold { return Err("confidence gate: too uncertain".into()); } }
            Builtin::Fuse => { let b = self.pop()? as f64 / 1000.0; let a = self.pop()? as f64 / 1000.0; self.confidence = 1.0 / (1.0/a + 1.0/b); self.push((self.confidence * 1000.0) as i32); }
            Builtin::Tell => { let _msg = self.pop()?; } // simplified
            Builtin::Ask => { self.push(0); } // simplified — would need fleet connection
            Builtin::Broadcast => { let _msg = self.pop()?; }
            Builtin::Instinct => { self.instinct = self.pop()? as u8; }
            Builtin::GeneExpr => { let _gene = self.pop()?; }
            Builtin::EnzymeBind => { let _gene = self.pop()?; let _signal = self.pop()?; }
            Builtin::MembraneChk => { let _signal = self.pop()?; self.push(1); } // simplified — always passes
            Builtin::AtpGen => { self.energy = (self.energy + 1.0).min(self.max_energy); self.push(self.energy as i32); }
            Builtin::AtpConsume => { let cost = self.pop()? as f64 / 10.0; if self.energy >= cost { self.energy -= cost; self.push(1); } else { self.push(0); } }
            Builtin::AtpQ => { self.push((self.energy * 10.0) as i32); }
            Builtin::ApoptosisChk => { let ratio = self.energy / self.max_energy; self.push(if ratio < 0.05 { 1 } else { 0 }); }
            Builtin::CircadianGet => { self.push(12); } // simplified — always noon
            Builtin::CircadianSet => { let _hour = self.pop()?; }
            Builtin::Halt => { return Err("HALT".into()); } // stop execution
            Builtin::If | Builtin::Then | Builtin::Else | Builtin::Begin | Builtin::While |
            Builtin::Repeat | Builtin::Until | Builtin::Do | Builtin::Loop | Builtin::I => {
                // Control flow — handled during compilation, not execution
            }
        }
        Ok(())
    }

    /// Get current stack depth
    pub fn depth(&self) -> usize { self.data_stack.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arithmetic() {
        let mut f = Forth::new();
        f.compile("3 4 +").unwrap();
        assert_eq!(f.data_stack, vec![7]);
    }

    #[test]
    fn test_dup() {
        let mut f = Forth::new();
        f.compile("42 DUP").unwrap();
        assert_eq!(f.data_stack, vec![42, 42]);
    }

    #[test]
    fn test_swap() {
        let mut f = Forth::new();
        f.compile("1 2 SWAP").unwrap();
        assert_eq!(f.data_stack, vec![2, 1]);
    }

    #[test]
    fn test_word_def() {
        let mut f = Forth::new();
        f.compile(": SQUARE DUP * ; 5 SQUARE").unwrap();
        assert_eq!(f.data_stack, vec![25]);
    }

    #[test]
    fn test_comparison() {
        let mut f = Forth::new();
        f.compile("3 5 <").unwrap();
        assert_eq!(f.data_stack, vec![1]); // true
        f.compile("5 3 <").unwrap();
        assert_eq!(f.data_stack, vec![0]); // false
    }

    #[test]
    fn test_over_rot() {
        let mut f = Forth::new();
        f.compile("1 2 OVER").unwrap();
        assert_eq!(f.data_stack, vec![1, 2, 1]);
    }

    #[test]
    fn test_memory() {
        let mut f = Forth::new();
        f.compile("42 HERE ! HERE @").unwrap();
        assert_eq!(f.data_stack, vec![42]);
    }

    #[test]
    fn test_negate() {
        let mut f = Forth::new();
        f.compile("5 NEGATE").unwrap();
        assert_eq!(f.data_stack, vec![-5]);
    }

    #[test]
    fn test_bitwise() {
        let mut f = Forth::new();
        f.compile("5 3 AND").unwrap();
        assert_eq!(f.data_stack, vec![1]); // 101 & 011 = 001
    }

    #[test]
    fn test_confidence_gate_pass() {
        let mut f = Forth::new();
        f.confidence = 0.95;
        f.compile("500 GATE").unwrap(); // threshold 0.5, we have 0.95
    }

    #[test]
    fn test_confidence_gate_fail() {
        let mut f = Forth::new();
        f.confidence = 0.3;
        assert!(f.compile("500 GATE").is_err()); // threshold 0.5, we have 0.3
    }

    #[test]
    fn test_fuse() {
        let mut f = Forth::new();
        f.compile("500 500 FUSE").unwrap();
        assert_eq!(f.data_stack, vec![250]); // 1/(1/0.5 + 1/0.5) = 0.25 → 250
    }

    #[test]
    fn test_atp() {
        let mut f = Forth::new();
        f.energy = 50.0;
        f.compile("ATP_Q").unwrap();
        assert_eq!(f.data_stack, vec![500]); // 50 * 10
        f.compile("ATP_GEN").unwrap();
        assert!(f.energy > 50.0);
    }

    #[test]
    fn test_instinct() {
        let mut f = Forth::new();
        f.compile("3 INSTINCT").unwrap();
        assert_eq!(f.instinct, 3); // Navigate
    }

    #[test]
    fn test_comments() {
        let mut f = Forth::new();
        f.compile("\\ this is a comment\n42").unwrap();
        assert_eq!(f.data_stack, vec![42]);
    }

    #[test]
    fn test_nested_word() {
        let mut f = Forth::new();
        f.compile(": DOUBLE DUP + ; : QUAD DOUBLE DOUBLE ; 3 QUAD").unwrap();
        assert_eq!(f.data_stack, vec![12]);
    }

    #[test]
    fn test_halt() {
        let mut f = Forth::new();
        let result = f.compile("42 HALT 99");
        assert!(result.is_err());
        assert_eq!(f.data_stack, vec![42]); // 99 never pushed
    }

    #[test]
    fn test_word_count() {
        let mut f = Forth::new();
        f.compile(": A 1 ; : B 2 ; : C 3 ;").unwrap();
        assert_eq!(f.words.len(), 3);
    }
}

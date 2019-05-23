fn main() {
    let mut vals = vec![2,1];
    for i in 2..10 {
        let update = iterate_dithering(&vals);
        println!("Length {}: {:?}", i, update);
        let (valid, invalid) = valid_invalid(&update);
        println!("Valid: {}, Invalid: {}", valid, invalid);
        println!("Total: {}, w/o Invalid: {}", valid + invalid, valid);
        println!("Percent Allowed: {:.3}%", 100. * ((valid as f64) / ((valid + invalid) as f64)));
        vals = update;
    }
}

type AbstractCount = HashMap<String, u64>

fn valid_invalid(vals: &AbstractCount) -> (u64, u64) {
    
    (vals.iter().take(4).sum(), vals.iter().skip(4).sum())
}

fn iterate_dithering(vals: &Vec<u64>) -> Vec<u64> {
    let mut new_vals = vec![0;vals.len() + 1];
    for (i,v) in vals.into_iter().enumerate() {
        if i % 2 == 0 {
            // even case
            new_vals[i+1] += v; // l
            new_vals[0] += 2*v; // r & f
        } else {
            // odd case
            new_vals[1] += v; // l
            new_vals[i+1] += v; // r
            new_vals[0] += v; // f
        }
    }
    new_vals
}

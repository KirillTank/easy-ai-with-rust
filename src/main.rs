#[derive(Clone)]
struct DynamicNet {
    pub weights: Vec<f64>,
    weights_per_input: usize,
}

impl DynamicNet {
    fn new(inputs_count: usize, weights_per_input: usize) -> Self {
        Self {
            weights: vec![0.1; inputs_count * weights_per_input],
            weights_per_input,
        }
    }

    fn predict(&self, inputs: &[f64]) -> f64 {
        inputs
            .iter()
            .zip(self.weights.chunks(self.weights_per_input))
            .map(|(&x, group)| x * group.iter().sum::<f64>())
            .sum()
    }

    fn train(&mut self, inputs: &[f64], target: f64) -> f64 {
        let current_pred = self.predict(inputs);
        let error = target - current_pred;

        let energy: f64 = inputs.iter().map(|x| x * x).sum();
        let mu = 0.5; // фактор устойчивости, обязателен для NLMS, должен быть в (0, 2), берём с запасом
        let lr = mu / (energy * self.weights_per_input as f64 + 1e-10);

        inputs
            .iter()
            .zip(self.weights.chunks_mut(self.weights_per_input))
            .for_each(|(&x, group)| {
                let delta = lr * error * x;
                group.iter_mut().for_each(|w| *w += delta);
            });

        target - self.predict(inputs)
    }

    fn start_train(&mut self, max_epoch: i32, dataset: &Vec<(Vec<f64>, f64)>) {
        for epoch in 0..max_epoch {
            let mut total_abs_error = 0.0;

            for (inputs, target) in dataset {
                let err = self.train(inputs, *target);
                total_abs_error += err.abs();
            }
            let avg_error = total_abs_error / dataset.len() as f64;

            if avg_error < 0.01 {
                println!("Эпоха {:>3}: Средняя |ошибка| = {:.6}", epoch, avg_error);
            }
            if avg_error < 0.00001 {
                println!("\nУспешная сходимость на эпохе {epoch}!");
                break;
            }
        }
    }

    fn print_res_train(&self, dataset: &[(Vec<f64>, f64)]) {
        println!("\n--- Итоговая проверка датасета ---");
        for (i, (inputs, target)) in dataset.iter().enumerate() {
            let pred = self.predict(inputs);
            let diff = (target - pred).abs();
            println!(
                "Пример {}: Цель = {:>5.1} | Итог сети = {:>8.4} | Погрешность = {:.4}",
                i + 1,
                target,
                pred,
                diff
            );
        }
    }
}

fn main() {
    let mut net = DynamicNet::new(5, 2);

    let dataset = vec![
        (vec![1.0, 2.0, 3.0, 0.5, 1.5], 100.0),
        (vec![2.0, 0.5, 1.0, 4.0, 0.0], 50.0),
        (vec![0.5, 1.5, 2.0, 1.0, 3.0], 80.0),
        (vec![3.0, 1.0, 0.5, 2.0, 2.0], 110.0),
    ];

    net.start_train(1000000, &dataset);
    net.print_res_train(&dataset);

    println!("1e-10: {}", 1e-10);
}

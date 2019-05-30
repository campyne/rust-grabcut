extern crate image;
extern crate rkm;

use image::*;
use rkm::*;

pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub w: u32,
    pub h: u32,
}

const COMPONENTS_COUNT: usize = 5;
const INIT_WITH_RECT: u8 = 0;
const INIT_WITH_MASK: u8 = 1;
const GC_BGD: u8 = 0;
const GC_FGD: u8 = 1;
const GC_PR_BGD: u8 = 2;
const GC_PR_FGD: u8 = 3;

type Color = [f64; 3];

#[derive(Default)]
struct GMM<'a> {
    pub _model: &'a mut [f64],
    //these are indexes into the model
    pub coefs: usize,
    pub mean: usize,
    pub cov: usize,

    pub inverse_covs: [[[f64; 3]; 3]; COMPONENTS_COUNT],
    pub cov_determs: [f64; COMPONENTS_COUNT],

    pub sums: [[f64; 3]; COMPONENTS_COUNT],
    pub prods: [[[f64; 3]; 3]; COMPONENTS_COUNT],
    pub sample_counts: [i64; COMPONENTS_COUNT],

    pub total_sample_count: i64,
}

impl<'a> GMM<'a> {
    fn new(model: &mut Vec<f64>) -> GMM {
        const MODEL_SIZE: usize = 3/*mean*/ + 9/*covariance*/ + 1/*component weight*/;
        if model.is_empty() {
            model.reserve(MODEL_SIZE * COMPONENTS_COUNT);
            //better way to created zeroed?
            for i in 0..MODEL_SIZE * COMPONENTS_COUNT {
                model.push(0.0);
            }
        }

        let mut gmm = GMM {
            _model: &mut model[0..],
            coefs: 0,
            mean: COMPONENTS_COUNT,
            cov: 3 * COMPONENTS_COUNT,
            ..Default::default()
        };

        for ci in 0..COMPONENTS_COUNT {
            if gmm._model[gmm.coefs + ci] > 0.0 {
                GMM::calc_inverse_cov_and_determ(&mut gmm, ci, 0.0);
            }
        }
        gmm
    }

    //these(w, which, which_component) need to be refactored or something
    fn w(&self, color: Color) -> f64 {
        let mut res = 0.0;
        for ci in 0..COMPONENTS_COUNT {
            res += self._model[self.coefs + ci] * self.which(ci, color);
        }
        res
    }

    fn which(&self, ci: usize, color: Color) -> f64 {
        let mut res = 0.0;
        if self._model[self.coefs + ci] > 0.0 {
            assert!(self.cov_determs[ci] > core::f64::EPSILON);
            let mut diff = color.clone();
            let mut m = self.mean + 3 * ci;
            diff[0] -= self._model[m];
            diff[1] -= self._model[m + 1];
            diff[2] -= self._model[m + 2];
            let mult = diff[0]
                   * (diff[0] * self.inverse_covs[ci][0][0]
                    + diff[1] * self.inverse_covs[ci][1][0]
                    + diff[2] * self.inverse_covs[ci][2][0])
                + diff[1]
                   * (diff[0] * self.inverse_covs[ci][0][1]
                    + diff[1] * self.inverse_covs[ci][1][1]
                    + diff[2] * self.inverse_covs[ci][2][1])
                + diff[2]
                   * (diff[0] * self.inverse_covs[ci][0][2]
                    + diff[1] * self.inverse_covs[ci][1][2]
                    + diff[2] * self.inverse_covs[ci][2][2]);
            res = 1.0 / self.cov_determs[ci].sqrt() * (-0.5 * mult).exp();
        }
        res
    }

    fn which_component(&self, color: Color) -> usize {
        let mut k = 0;
        let mut max = 0.0;

        for ci in 0..COMPONENTS_COUNT {
            let p = self.which(ci, color);
            if p > max {
                k = ci;
                max = p;
            }
        }
        k
    }

    fn init_learning(&mut self) {
        for ci in 0..COMPONENTS_COUNT {
            self.sums[ci][0] = 0.0;
            self.sums[ci][1] = 0.0;
            self.sums[ci][2] = 0.0;
            self.prods[ci][0][0] = 0.0;
            self.prods[ci][0][1] = 0.0;
            self.prods[ci][0][2] = 0.0;
            self.prods[ci][1][0] = 0.0;
            self.prods[ci][1][1] = 0.0;
            self.prods[ci][1][2] = 0.0;
            self.prods[ci][2][0] = 0.0;
            self.prods[ci][2][1] = 0.0;
            self.prods[ci][2][2] = 0.0;
            self.sample_counts[ci] = 0;
        }
        self.total_sample_count = 0;
    }

    fn add_sample(&mut self, ci: usize, color: Color) {
        self.sums[ci][0] += color[0];
        self.sums[ci][1] += color[1];
        self.sums[ci][2] += color[2];
        self.prods[ci][0][0] += color[0] * color[0];
        self.prods[ci][0][1] += color[0] * color[1];
        self.prods[ci][0][2] += color[0] * color[2];
        self.prods[ci][1][0] += color[1] * color[0];
        self.prods[ci][1][1] += color[1] * color[1];
        self.prods[ci][1][2] += color[1] * color[2];
        self.prods[ci][2][0] += color[2] * color[0];
        self.prods[ci][2][1] += color[2] * color[1];
        self.prods[ci][2][2] += color[2] * color[2];
        self.sample_counts[ci] += 1;
        self.total_sample_count += 1;
    }

    fn end_learning(&mut self) {
        for ci in 0..COMPONENTS_COUNT {
            let n = self.sample_counts[ci] as f64;
            if n == 0.0 {
                self._model[self.coefs + ci] = 0.0;
            } else {
                assert!(self.total_sample_count > 0);
                let inv_n = 1.0 / n;
                self._model[self.coefs + ci] = n / self.total_sample_count as f64;

                let m = self.mean + 3 * ci;
                self._model[m] = self.sums[ci][0] * inv_n;
                self._model[m + 1] = self.sums[ci][1] * inv_n;
                self._model[m + 2] = self.sums[ci][2] * inv_n;

                let c = self.cov + 9 * ci;
                self._model[c] = self.prods[ci][0][0] * inv_n - self._model[m] * self._model[m];
                self._model[c + 1] =
                    self.prods[ci][0][1] * inv_n - self._model[m] * self._model[m + 1];
                self._model[c + 2] =
                    self.prods[ci][0][2] * inv_n - self._model[m] * self._model[m + 2];
                self._model[c + 3] =
                    self.prods[ci][1][0] * inv_n - self._model[m + 1] * self._model[m];
                self._model[c + 4] =
                    self.prods[ci][1][1] * inv_n - self._model[m + 1] * self._model[m + 1];
                self._model[c + 5] =
                    self.prods[ci][1][2] * inv_n - self._model[m + 1] * self._model[m + 2];
                self._model[c + 6] =
                    self.prods[ci][2][0] * inv_n - self._model[m + 2] * self._model[m];
                self._model[c + 7] =
                    self.prods[ci][2][1] * inv_n - self._model[m + 2] * self._model[m + 1];
                self._model[c + 8] =
                    self.prods[ci][2][2] * inv_n - self._model[m + 2] * self._model[m + 2];

                GMM::calc_inverse_cov_and_determ(self, ci, 0.01);
            }
        }
    }

    fn calc_inverse_cov_and_determ(gmm: &mut GMM, ci: usize, singular_fix: f64) {
        if gmm._model[gmm.coefs + ci] > 0.0 {
            //stuff
            let c = &mut gmm._model[gmm.cov + 9 * ci..];
            let mut dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6])
                + c[2] * (c[3] * c[7] - c[4] * c[6]);
            if dtrm <= 1e-6 && singular_fix > 0.0 {
                c[0] += singular_fix;
                c[4] += singular_fix;
                c[8] += singular_fix;
                dtrm = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6])
                    + c[2] * (c[3] * c[7] - c[4] * c[6]);
            }
            gmm.cov_determs[ci] = dtrm;

            assert!(dtrm > core::f64::EPSILON);
            let inv_dtrm = 1.0 / dtrm;
            gmm.inverse_covs[ci][0][0] =  (c[4] * c[8] - c[5] * c[7]) * inv_dtrm;
            gmm.inverse_covs[ci][1][0] = -(c[3] * c[8] - c[5] * c[6]) * inv_dtrm;
            gmm.inverse_covs[ci][2][0] =  (c[3] * c[7] - c[4] * c[6]) * inv_dtrm;
            gmm.inverse_covs[ci][0][1] = -(c[1] * c[8] - c[2] * c[7]) * inv_dtrm;
            gmm.inverse_covs[ci][1][1] =  (c[0] * c[8] - c[2] * c[6]) * inv_dtrm;
            gmm.inverse_covs[ci][2][1] = -(c[0] * c[7] - c[1] * c[6]) * inv_dtrm;
            gmm.inverse_covs[ci][0][2] =  (c[1] * c[5] - c[2] * c[4]) * inv_dtrm;
            gmm.inverse_covs[ci][1][2] = -(c[0] * c[5] - c[2] * c[3]) * inv_dtrm;
            gmm.inverse_covs[ci][2][2] =  (c[0] * c[4] - c[1] * c[3]) * inv_dtrm;
        }
    }
}

fn dot(a: Color, b: Color) -> f64 {
    //compute dot product of a and b
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

//beta = 1/(2*avg(sqr(||color pixel_i - color pixel_j||)))
fn calc_beta(img: &image::DynamicImage) -> f64 {
    let mut beta = 0.0;

    //TODO hideous, better way?
    for (x, y, pixel) in img.pixels() {
        let color: Color = {
            [
                pixel.data[0] as f64,
                pixel.data[1] as f64,
                pixel.data[2] as f64,
            ]
        };
        if x > 0 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x - 1, y).data[0] as f64),
                    (color[1] - img.get_pixel(x - 1, y).data[1] as f64),
                    (color[2] - img.get_pixel(x - 1, y).data[2] as f64),
                ]
            };
            beta += dot(diff, diff);
        }
        if y > 0 && x > 0 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x - 1, y - 1).data[0] as f64),
                    (color[1] - img.get_pixel(x - 1, y - 1).data[1] as f64),
                    (color[2] - img.get_pixel(x - 1, y - 1).data[2] as f64),
                ]
            };
            beta += dot(diff, diff);
        }
        if y > 0 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x, y - 1).data[0] as f64),
                    (color[1] - img.get_pixel(x, y - 1).data[1] as f64),
                    (color[2] - img.get_pixel(x, y - 1).data[2] as f64),
                ]
            };
            beta += dot(diff, diff);
        }
        if y > 0 && x < img.width() - 1 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x + 1, y - 1).data[0] as f64),
                    (color[1] - img.get_pixel(x + 1, y - 1).data[1] as f64),
                    (color[2] - img.get_pixel(x + 1, y - 1).data[2] as f64),
                ]
            };
            beta += dot(diff, diff);
        }
    }
    if beta <= core::f64::EPSILON {
        beta = 0.0;
    } else {
        beta = 1.0
            / (2.0 * beta
                / (4 * img.width() * img.height() - 3 * img.width() - 3 * img.width() + 2) as f64);
    }
    beta
}

fn calc_weights(
    img: &image::DynamicImage,
    beta: f64,
    gamma: f64,
) -> (Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut left_weight = vec![vec![0.0; img.height() as usize]; img.width() as usize];
    let mut upper_left_weight = vec![vec![0.0; img.height() as usize]; img.width() as usize];
    let mut upper_weight = vec![vec![0.0; img.height() as usize]; img.width() as usize];
    let mut upper_right_weight = vec![vec![0.0; img.height() as usize]; img.width() as usize];
    let gamma_div_sqrt2 = gamma / 2.0_f64.sqrt();

    for (x, y, pixel) in img.pixels() {
        let color: Color = {
            [
                pixel.data[0] as f64,
                pixel.data[1] as f64,
                pixel.data[2] as f64,
            ]
        };
        if x > 0 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x - 1, y).data[0] as f64),
                    (color[1] - img.get_pixel(x - 1, y).data[1] as f64),
                    (color[2] - img.get_pixel(x - 1, y).data[2] as f64),
                ]
            };
            left_weight[x as usize][y as usize] = gamma * (-beta * dot(diff, diff)).exp();
        } else {
            left_weight[x as usize][y as usize] = 0.0;
        }
        if y > 0 && x > 0 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x - 1, y - 1).data[0] as f64),
                    (color[1] - img.get_pixel(x - 1, y - 1).data[1] as f64),
                    (color[2] - img.get_pixel(x - 1, y - 1).data[2] as f64),
                ]
            };
            upper_left_weight[x as usize][y as usize] = gamma * (-beta * dot(diff, diff)).exp();
        } else {
            upper_left_weight[x as usize][y as usize] = 0.0;
        }
        if y > 0 {
            let diff = {
                [
                    (color[0] - img.get_pixel(x, y - 1).data[0] as f64),
                    (color[1] - img.get_pixel(x, y - 1).data[1] as f64),
                    (color[2] - img.get_pixel(x, y - 1).data[2] as f64),
                ]
            };
            upper_weight[x as usize][y as usize] = gamma * (-beta * dot(diff, diff)).exp();
        } else {
            upper_weight[x as usize][y as usize] = 0.0;
        }
        if y > 0 && x + 1 < img.width() {
            let diff = {
                [
                    (color[0] - img.get_pixel(x + 1, y - 1).data[0] as f64),
                    (color[1] - img.get_pixel(x + 1, y - 1).data[1] as f64),
                    (color[2] - img.get_pixel(x + 1, y - 1).data[2] as f64),
                ]
            };
            upper_right_weight[x as usize][y as usize] = gamma * (-beta * dot(diff, diff)).exp();
        } else {
            upper_right_weight[x as usize][y as usize] = 0.0;
        }
    }
    (
        left_weight,
        upper_left_weight,
        upper_weight,
        upper_right_weight,
    )
}

fn check_mask(img: &image::DynamicImage, mask: &image::GrayImage) {
    assert!(img.dimensions() == mask.dimensions());
    for (x, y, pixel) in mask.enumerate_pixels() {
        //pixels in the mask must be one of the correct values to indicate
        //foreground, background, probably foreground, or probably background
        assert!(pixel.data[0] == GC_BGD
             || pixel.data[0] == GC_FGD
             || pixel.data[0] == GC_PR_BGD
             || pixel.data[0] == GC_PR_FGD
        );
    }
}

fn init_mask_with_rect(mask: &mut image::GrayImage, img_size: (u32, u32), rect: Rect) {
    //TODO check rect is a subsection of the image, i.e. x,y,w,h fits within
    for (x, y, pixel) in mask.enumerate_pixels_mut() {
        //if the pixel is in the rectangle set it to probably foreground
        if x > rect.x && x < rect.x + rect.w &&//TODO better way to do this, maybe rect as a tuple of points?
            y > rect.y && y < rect.y + rect.h
        {
            *pixel = image::Luma([GC_PR_FGD]);
        } else {
            //otherwise set it to background
            *pixel = image::Luma([GC_BGD]);
        }
    }
}

fn init_gmms(
    img: &image::DynamicImage,
    mask: &image::GrayImage,
    bgd_gmm: &mut GMM,
    fgd_gmm: &mut GMM,
) {
    let mut bgd_labels: Vec<usize> = Vec::new();
    let mut fgd_labels: Vec<usize> = Vec::new();
    let mut bgd_samples: Vec<Color> = Vec::new();
    let mut fgd_samples: Vec<Color> = Vec::new();

    //I need to figure out how to set this all up so usable with rkm::kmeans_lloyd
    //or use a different kmeans implementation
    //
    //iterate through mask and add color samples from image to bgd or fgd
    //depending on value of mask pixels
    for (x, y, pixel) in mask.enumerate_pixels() {
        let color: Color = {
            [
                img.get_pixel(x, y).data[0] as f64,
                img.get_pixel(x, y).data[1] as f64,
                img.get_pixel(x, y).data[2] as f64,
            ]
        };
        if pixel.data[0] == GC_BGD || pixel.data[0] == GC_PR_BGD {
            bgd_samples.push(color);
        } else {
            //GC_PR_FGD || GC_FGD
            fgd_samples.push(color);
        }
    }

    //do kmeans
    //bgd_labels = rkm::kmeans_lloyd(&bgd_samples, COMPONENTS_COUNT);
    //fgd_labels = rkm::kmeans_lloyd(&fgd_samples, COMPONENTS_COUNT);

    bgd_gmm.init_learning();
    //add samples to gmm
    for i in 0..bgd_samples.len() {
        //bgd_gmm.add_sample(bgd_labels[i], bgd_samples[i]);
    }
    bgd_gmm.end_learning();

    fgd_gmm.init_learning();
    //add samples to gmm
    for i in 0..fgd_samples.len() {
        //fgd_gmm.add_sample(fgd_labels[i], fgd_samples[i]);
    }
    fgd_gmm.end_learning();
}

fn assign_gmms_components(
    img: &image::DynamicImage,
    mask: &image::GrayImage,
    bgd_gmm: &GMM,
    fgd_gmm: &GMM,
    comp_idxs: &mut Vec<Vec<usize>>,
) {
    for (x, y, pixel) in img.pixels() {
        let color: Color = {
            [
                pixel.data[0] as f64,
                pixel.data[1] as f64,
                pixel.data[2] as f64,
            ]
        };
        comp_idxs[x as usize][y as usize] = {
            if mask.get_pixel(x, y).data[0] == GC_BGD || mask.get_pixel(x, y).data[0] == GC_PR_BGD {
                bgd_gmm.which_component(color)
            } else {
                fgd_gmm.which_component(color)
            }
        };
    }
}

fn learn_gmms(
    img: &image::DynamicImage,
    mask: &image::GrayImage,
    comp_idxs: &Vec<Vec<usize>>,
    bgd_gmm: &mut GMM,
    fgd_gmm: &mut GMM,
) {
    bgd_gmm.init_learning();
    fgd_gmm.init_learning();

    for ci in 0..COMPONENTS_COUNT {
        for (x, y, pixel) in img.pixels() {
            if comp_idxs[x as usize][y as usize] == ci {
                let color: Color = {
                    [
                        pixel.data[0] as f64,
                        pixel.data[1] as f64,
                        pixel.data[2] as f64,
                    ]
                };

                if mask.get_pixel(x, y).data[0] == GC_BGD
                    || mask.get_pixel(x, y).data[0] == GC_PR_BGD
                {
                    bgd_gmm.add_sample(ci, color);
                } else {
                    fgd_gmm.add_sample(ci, color);
                }
            }
        }
    }
    bgd_gmm.end_learning();
    fgd_gmm.end_learning();
}

fn construct_gc_graph() {
    //need to figure what graph/maxflow/mincut library stuff to use
}

fn estimate_segmentation() {}

pub fn grabcut<'a>(
    img: &image::DynamicImage,
    mask: &'a mut image::GrayImage,
    rect: Rect,
    bgd_model: &'a mut Vec<f64>,
    fgd_model: &'a mut Vec<f64>,
    iter_count: u32,
    mode: u8,
) {
    //check img for errors?

    //initialize gaussian models for foreground and background
    let mut fgd_gmm = GMM::new(fgd_model);
    let mut bgd_gmm = GMM::new(bgd_model);

    let mut comp_idxs = vec![vec![0; img.height() as usize]; img.width() as usize];

    //Initialization
    //
    //If using a rectangle the background set to pixels outside of rect.
    //The area inside the rectangle is set to probably foreground.
    //If rerunning with a mask from previous run check the mask is OK
    if mode == INIT_WITH_RECT || mode == INIT_WITH_MASK {
        if mode == INIT_WITH_RECT {
            init_mask_with_rect(mask, img.dimensions(), rect);
        }
        if mode == INIT_WITH_MASK {
            check_mask(img, mask);
        }
        init_gmms(img, mask, &mut bgd_gmm, &mut fgd_gmm);
    }
    let gamma: f64 = 50.0;
    let lambda: f64 = 9.0 * gamma;
    let beta = calc_beta(img);

    //calculate the weights for neighboring pixels
    let (left_weight, upper_left_weight, upper_weight, upper_right_weight) =
        calc_weights(img, beta, gamma);

    for i in 0..iter_count {
        //Assign GMM components to pixels
        assign_gmms_components(&img, &mask, &bgd_gmm, &fgd_gmm, &mut comp_idxs);
        //Learn GMM parameters
        learn_gmms(img, mask, &comp_idxs, &mut bgd_gmm, &mut fgd_gmm);
        //construct the graph using the weights etc
        construct_gc_graph();
        //Estimate segmentation
        estimate_segmentation();
    }
}

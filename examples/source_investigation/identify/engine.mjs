/** ICRS catalog associations. No classification, discovery, or probability claims. */
export const VERSION = '0.1.0';
const RAD = Math.PI / 180, ARCSEC = 3600 / RAD;
function number(v, name, min = -Infinity, max = Infinity) {
  if (typeof v !== 'number' || !Number.isFinite(v) || v < min || v > max) throw new TypeError(`${name} must be a finite number in [${min}, ${max}]`);
  return v;
}
function text(v, name) {
  if (typeof v !== 'string' || !v.trim()) throw new TypeError(`${name} must be a nonempty string`);
  return v;
}
function position(o, name) {
  if (!o || typeof o !== 'object' || Array.isArray(o)) throw new TypeError(`${name} must be an object`);
  number(o.ra_deg, `${name}.ra_deg`, 0, 360);
  if (o.ra_deg === 360) throw new TypeError(`${name}.ra_deg must be less than 360`);
  number(o.dec_deg, `${name}.dec_deg`, -90, 90);
  number(o.epoch_year, `${name}.epoch_year`, 1800, 2200);
  if (o.sigma_arcsec !== undefined) number(o.sigma_arcsec, `${name}.sigma_arcsec`, Number.MIN_VALUE, 3600);
  if (o.flags !== undefined && (!Array.isArray(o.flags) || o.flags.some(f => typeof f !== 'string'))) throw new TypeError(`${name}.flags must be strings`);
}
/** Stable great-circle angle, including RA wrap and the poles. */
export function separationArcsec(a, b) {
  const d1 = a.dec_deg * RAD, d2 = b.dec_deg * RAD, da = (b.ra_deg - a.ra_deg) * RAD;
  const y = Math.hypot(Math.cos(d2) * Math.sin(da), Math.cos(d1) * Math.sin(d2) - Math.sin(d1) * Math.cos(d2) * Math.cos(da));
  const x = Math.sin(d1) * Math.sin(d2) + Math.cos(d1) * Math.cos(d2) * Math.cos(da);
  return Math.atan2(y, x) * ARCSEC;
}
/** Approximate constant tangential motion, mu_alpha* includes cos(dec).
 * Excludes parallax, radial velocity, acceleration, and Solar System ephemerides.
 */
export function propagate(row, epoch) {
  const dt = epoch - row.epoch_year;
  if (dt === 0) return { ...row, propagation: 'same_epoch' };
  if (row.pmra_mas_yr === undefined || row.pmdec_mas_yr === undefined) return null;
  const a = row.ra_deg * RAD, d = row.dec_deg * RAD;
  const east = row.pmra_mas_yr * dt / 1000 / ARCSEC;
  const north = row.pmdec_mas_yr * dt / 1000 / ARCSEC;
  const angle = Math.hypot(east, north);
  // Deliberately limited approximation. Outside this range require external astrometry.
  if (Math.abs(dt) > 50 || angle > .01) return null;
  const v = [Math.cos(d) * Math.cos(a), Math.cos(d) * Math.sin(a), Math.sin(d)];
  const e = [-Math.sin(a), Math.cos(a), 0], n = [-Math.sin(d) * Math.cos(a), -Math.sin(d) * Math.sin(a), Math.cos(d)];
  const k = angle === 0 ? 1 : Math.sin(angle) / angle;
  const u = v.map((x, i) => x * Math.cos(angle) + k * (east * e[i] + north * n[i]));
  return { ...row, ra_deg: (Math.atan2(u[1], u[0]) / RAD + 360) % 360,
    dec_deg: Math.atan2(u[2], Math.hypot(u[0], u[1])) / RAD,
    epoch_year: epoch, propagation: 'approximate_proper_motion' };
}
/** All candidates inside an angular cone are retained; proximity is not identity. */
export function identify(detections, catalog, options = {}) {
  if (!Array.isArray(detections) || !Array.isArray(catalog)) throw new TypeError('detections and catalog must be arrays');
  const config = { radius_arcsec: 6, max_normalized_separation: 3, ...options };
  for (const key of Object.keys(config)) if (!['radius_arcsec', 'max_normalized_separation'].includes(key)) throw new TypeError(`Unknown option: ${key}`);
  number(config.radius_arcsec, 'radius_arcsec', Number.MIN_VALUE, 3600);
  number(config.max_normalized_separation, 'max_normalized_separation', Number.MIN_VALUE, 10);
  const seen = new Set();
  detections.forEach((d, i) => {
    position(d, `detections[${i}]`); text(d.id, 'detection.id');
    if (seen.has(d.id)) throw new TypeError(`Duplicate detection ID: ${d.id}`);
    seen.add(d.id);
  });
  const ids = new Set();
  catalog.forEach((r, i) => {
    position(r, `catalog[${i}]`); text(r.id, 'catalog.id'); text(r.catalog, 'catalog.catalog'); text(r.source_url, 'catalog.source_url');
    if (!/^https?:\/\//.test(r.source_url)) throw new TypeError('catalog.source_url must use HTTP(S)');
    const key = JSON.stringify([r.catalog, r.id]);
    if (ids.has(key)) throw new TypeError(`Duplicate catalog entry: ${key}`);
    ids.add(key);
    for (const p of ['pmra_mas_yr', 'pmdec_mas_yr']) if (r[p] !== undefined) number(r[p], p);
    if ((r.pmra_mas_yr === undefined) !== (r.pmdec_mas_yr === undefined)) throw new TypeError('Supply both proper-motion components');
  });
  const results = detections.map(d => {
    const candidates = [], skipped = [];
    for (const r of catalog) {
      const p = propagate(r, d.epoch_year);
      if (!p) { skipped.push({ catalog: r.catalog, id: r.id, reason: 'epoch_not_comparable' }); continue; }
      const distance = separationArcsec(d, p);
      if (distance > config.radius_arcsec) continue;
      // Reusing a catalog position error at a new epoch is not defensible without PM covariance.
      const sigma = p.propagation === 'same_epoch' && d.sigma_arcsec !== undefined && r.sigma_arcsec !== undefined
        ? Math.hypot(d.sigma_arcsec, r.sigma_arcsec) : null;
      candidates.push({ catalog: r.catalog, id: r.id, label: r.label ?? r.id,
        catalog_type: r.object_type ?? null, source_url: r.source_url,
        separation_arcsec: distance, normalized_separation: sigma === null ? null : distance / sigma,
        propagated_position: { ra_deg: p.ra_deg, dec_deg: p.dec_deg, epoch_year: d.epoch_year },
        propagation: p.propagation, flags: r.flags ?? [] });
    }
    candidates.sort((a, b) => a.separation_arcsec - b.separation_arcsec || a.catalog.localeCompare(b.catalog) || a.id.localeCompare(b.id));
    const reasons = [], warnings = [], best = candidates[0];
    let status;
    if (!catalog.length) { status = 'inconclusive'; reasons.push('empty_catalog'); }
    else if (d.flags?.length) { status = 'inconclusive'; reasons.push('detection_quality_flags'); }
    else if (skipped.length) { status = 'inconclusive'; reasons.push('catalog_epoch_coverage_incomplete'); }
    else if (!candidates.length) { status = 'no_match'; reasons.push('no_counterpart_within_search_radius'); }
    else if (candidates.length > 1) { status = 'ambiguous'; reasons.push('multiple_counterparts_within_search_radius'); }
    else if (best.flags.length) { status = 'inconclusive'; reasons.push('catalog_quality_flags'); }
    else if (best.normalized_separation === null) { status = 'inconclusive'; reasons.push('comparable_position_uncertainties_required'); }
    else if (best.normalized_separation > config.max_normalized_separation) { status = 'inconclusive'; reasons.push('position_inconsistent_with_reported_uncertainty'); }
    else { status = 'tentative_association'; reasons.push('single_positionally_consistent_counterpart'); }
    if (candidates.some(c => c.propagation === 'approximate_proper_motion')) warnings.push('Approximate proper motion only; uncertainty propagation, parallax and radial velocity are not modeled.');
    return { detection_id: d.id, position: { ra_deg: d.ra_deg, dec_deg: d.dec_deg, epoch_year: d.epoch_year },
      status, reasons, flags: d.flags ?? [], candidates, skipped, warnings,
      identity_confirmed: false, discovery_claim: false };
  });
  // Independent detections can share a counterpart. Flag this; do not silently merge them.
  const links = new Map();
  for (const r of results) if (r.status === 'tentative_association') {
    const c = r.candidates[0], key = JSON.stringify([c.catalog, c.id]);
    if (!links.has(key)) links.set(key, []); links.get(key).push(r);
  }
  for (const rows of links.values()) if (rows.length > 1) for (const r of rows) {
    r.warnings.push('Counterpart is shared by multiple detections; blending or repeated exposures require review.');
  }
  return { schema_version: '1.0', software: { name: 'searchx-identify', version: VERSION },
    frame: 'ICRS', config, summary: results.reduce((s, r) => { s[r.status] = (s[r.status] ?? 0) + 1; return s; }, {}),
    limitations: ['Catalog associations are tentative, never independent validation.',
      'No-match applies only to the supplied catalog and search radius; it is not a discovery.',
      'No FITS source extraction, orbital fitting, live catalog lookup, or REALM verification is performed.'], results };
}

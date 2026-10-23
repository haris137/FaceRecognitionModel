# Consent process and data collection guide

## 1. Consent form (put this at the top of the Google Form)

**Title:** Face photos for the NED TechFest "Nemo's Deep Dive" face recognition research project

> We are a student team at NED University building a face recognition system that works fairly on Pakistani faces, for the AI Alliance TechFest competition (October 2026).
>
> **What we ask:** 10–20 photos of your face and one short video (5–10 s), plus your gender, age group, province/region and (optionally) skin tone.
>
> **How it is used:** only to train and test our model for this competition and its presentation. Faces in slides are shown only if you tick the separate box below.
>
> **How it is stored:** in a private Google Drive folder accessible only to our team. You get a participant code (e.g. P017); your name is never stored with the images.
>
> **Not shared:** we will not publish, sell or give your photos to anyone, including the organisers' public materials.
>
> **Deletion:** all photos and the trained model will be deleted by **30 November 2026**. You can withdraw at any time before then by messaging [team contact] with your participant code; we delete your folder and retrain.
>
> **Voluntary:** taking part is optional and you can skip any question. You must be **18 or older**.

Required checkboxes:
- [ ] I am 18 or older.
- [ ] I have read the above and **I consent** to my photos being used as described.
- [ ] (optional) My face may appear in the competition presentation slides.

Keep the response sheet (timestamp + participant code + consent) as your consent log. Mention its existence, not its contents, in the report.

## 2. Form fields (match the importer's defaults)

| Field | Type | Values |
|---|---|---|
| Participant code | short text | P001 … P200 (hand them out yourselves) |
| I consent | checkbox | "Yes, I consent" |
| Gender | choice | male / female / prefer not to say |
| Age group | choice | 18-25 / 26-35 / 36-50 / 51+ |
| Province / region | choice | Sindh / Punjab / KPK / Balochistan / GB / AJK / Islamabad / other |
| Skin tone | choice (optional) | light / medium / dark (self-described) |
| Photos upload | file upload, up to 10 files, 10 MB each | images |
| More photos upload | file upload | images |
| Video upload | file upload, 1 file, 100 MB | video |

Then run `scripts/import_google_form.py` (only consenting rows are imported).

## 3. Shot list for each volunteer (send this with the form)

Aim for variety; this is what makes the model robust.

1. 3 normal selfies, different days or places if possible
2. 1 photo in **low light** (room lights off, phone screen light only)
3. 1 photo **outdoors** in sunlight
4. 1 photo **from a distance** (someone else takes it, ~3 m away) — mimics CCTV
5. 1 photo with **head covering** you normally wear (dupatta, hijab, cap, turban) if any
6. 1 photo **with glasses / sunglasses** if you have them
7. 1 photo **with a face mask**
8. 1 slightly **side-angled** photo (~30°)
9. 1 photo from an **old/low-quality** phone, or a WhatsApp-forwarded copy (compressed)
10. Video: slowly turn head left → right, 5–10 seconds

Rename files with condition tags when you can, e.g. `img5__hijab-lowlight.jpg`. Tags used in the group analysis:
`hijab dupatta turban cap scarf | glasses sunglasses | mask | beard moustache | lowlight dark backlight | cctv lowres blur | outdoor`.
Alternatively fill `raw/image_tags.csv` with columns `identity,file,tags` (tags separated by `|`).

## 4. Balance targets for 200 people

- Gender: as close to 50/50 as possible.
- Age: at least 15% outside 18–25 (ask family members).
- Region: deliberately recruit from all provinces, not only Karachi.
- Conditions: at least 60 people with a head covering photo, 60 with glasses, everyone with a mask photo.

## 5. What to say in the presentation

- Consent was opt-in, written, adults only, purpose-limited, with a deletion date and a withdrawal route.
- Data was pseudonymised (participant codes), stored privately, never redistributed.
- Public dataset (halfFace) used under its licence and cited; used for training only.
- Test identities never appear in training (identity-disjoint split, verified by an assert in code).

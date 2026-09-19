```css
<style>

.stApp {
    background: linear-gradient(
        135deg,
        #fff7ed 0%,
        #ffedd5 50%,
        #fef3c7 100%
    );
    color: #292524;
}

.block-container {
    max-width: 1250px;
    padding-top: 45px;
    padding-bottom: 60px;
}


/* =========================================
   MAIN TITLE
   ========================================= */

.main-title {
    font-size: 48px;
    font-weight: 800;
    color: #292524;
    margin-bottom: 12px;
    letter-spacing: -1px;
}

.subtitle {
    color: #78716c;
    font-size: 17px;
    margin-bottom: 35px;
}


/* =========================================
   LEFT PANEL
   ========================================= */

.left-box {
    background-color: rgba(255, 255, 255, 0.92);
    border: 1px solid #fed7aa;
    border-radius: 24px;
    padding: 28px;
    box-shadow: 0 10px 30px rgba(120, 53, 15, 0.10);
}

.left-heading {
    font-size: 32px;
    font-weight: 800;
    color: #292524;
    margin-top: 24px;
    margin-bottom: 12px;
}

.left-text {
    color: #57534e;
    font-size: 15px;
    line-height: 1.8;
    margin-bottom: 25px;
}


/* =========================================
   LEFT MAIN FOOD IMAGE
   ========================================= */

.left-box img {
    width: 100%;
    height: 390px;
    object-fit: cover;
    border-radius: 18px;
}


/* =========================================
   RIGHT SIDE
   ========================================= */

.recommend-title {
    font-size: 32px;
    font-weight: 800;
    color: #292524;
    margin-bottom: 8px;
}

.recommend-subtitle {
    color: #78716c;
    font-size: 15px;
    margin-bottom: 28px;
}


/* =========================================
   RESTAURANT DETAILS
   ========================================= */

.restaurant-name {
    font-size: 20px;
    font-weight: 700;
    color: #292524;
    margin-bottom: 8px;
}

.restaurant-info {
    color: #57534e;
    font-size: 14px;
    margin: 7px 0;
}

.match {
    color: #ea580c;
    font-weight: 700;
    font-size: 15px;
    margin-top: 10px;
}


/* =========================================
   BUTTON
   ========================================= */

div.stButton > button {
    width: 100%;
    height: 54px;
    border-radius: 14px;
    border: none;
    background: linear-gradient(
        90deg,
        #f97316,
        #ea580c
    );
    color: white;
    font-size: 17px;
    font-weight: 700;
    margin-top: 8px;
    margin-bottom: 25px;
}

div.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #ea580c,
        #c2410c
    );
    color: white;
}


/* =========================================
   ALL FOOD IMAGES
   ========================================= */

div[data-testid="stImage"] img {
    border-radius: 16px;
}


/* =========================================
   RECOMMENDATION SPACING
   ========================================= */

div[data-testid="stHorizontalBlock"] {
    margin-bottom: 10px;
}


/* =========================================
   MOBILE
   ========================================= */

@media (max-width: 768px) {

    .main-title {
        font-size: 36px;
    }

    .left-heading {
        font-size: 26px;
    }

    .left-box img {
        height: 280px;
    }

    .recommend-title {
        font-size: 26px;
    }

}

</style>
```

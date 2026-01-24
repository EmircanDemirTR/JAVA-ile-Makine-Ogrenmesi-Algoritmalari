plugins {
    id("java")
}

// Proje Bilgileri

group = "com.btkakademi"
version = "1.0-SNAPSHOT"
description = "BTK Akademi - Java ML Algoritmalari Egitimi"

repositories {
    mavenCentral()
}

// Bağımlılıklar
dependencies {
    implementation("com.github.haifengl:smile-core:5.1.0")
    implementation("com.github.haifengl:smile-plot:5.1.0")

    implementation("nz.ac.waikato.cms.weka:weka-stable:3.8.6")

    implementation("org.slf4j:slf4j-simple:2.0.16")
}

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

tasks.withType<JavaCompile> {
    options.encoding = "UTF-8"
}
